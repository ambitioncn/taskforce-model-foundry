// Durable CLI/host training request, outside the caller's process lifetime.
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadConfig } from '../config.mjs';
import { ensureDir, exists, readJsonIfExists, writeJson } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { status as trainingStatus } from '../api.mjs';
import { launchBackgroundWorker, requireJobId } from './async-evaluate.mjs';

const workerFile = fileURLToPath(new URL('./async-train-worker.mjs', import.meta.url));

async function location({ baseDir, configFile, jobId, requestId }) {
  const { file, config } = await loadConfig({ file: configFile, baseDir });
  const root = path.join(config.runRootAbsolute, requireJobId(jobId), 'background-training');
  const dir = path.join(root, requestId);
  return { file, root, dir, statusFile: path.join(dir, 'status.json') };
}

export async function startBackgroundTraining({
  baseDir, configFile, jobId, resume = true, resumeCancelled = false, auditEveryStep = 10, retry = false,
} = {}) {
  requireJobId(jobId);
  const { file } = await loadConfig({ file: configFile, baseDir });
  const device = process.env.TMF_DEVICE ?? 'cpu';
  if (!['cpu', 'cuda'].includes(device)) throw new Error('unsupported TMF_DEVICE=' + device);
  const request = {
    kind: 'train', configFile: file, configDigest: await sha256File(file), jobId,
    resume, resumeCancelled, auditEveryStep, device, pythonBin: process.env.TMF_PYTHON ?? 'python3',
    hfMaxMemoryMb: process.env.TMF_HF_MAX_MEMORY_MB ?? null,
  };
  // Resume-cancelled changes the next attempt, not the identity of the job.
  // Otherwise the job-level ownership lock would reject a legitimate resume.
  const requestId = sha256Json({ ...request, resumeCancelled: undefined }).slice(0, 24);
  const loc = await location({ baseDir, configFile: file, jobId, requestId });
  await ensureDir(loc.root);
  // A jobId is immutable, so it belongs to one background request for its
  // lifetime. mkdir is atomic across concurrent callers; changed config or
  // device needs a new jobId rather than racing two workers on the same state.
  const activeDir = path.join(loc.root, 'active.lock');
  const activeFile = path.join(activeDir, 'request.json');
  try {
    await mkdir(activeDir);
    await writeJson(activeFile, { requestId });
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    const active = await readJsonIfExists(activeFile);
    if (!active?.requestId) throw new Error('background training job launch is initializing; retry status shortly');
    if (active.requestId !== requestId) throw new Error(`training job ${jobId} is bound to request ${active.requestId}; use a new jobId for changed inputs`);
  }
  let attempt = 1;
  try {
    await mkdir(loc.dir);
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    const prior = await backgroundTrainingStatus({ baseDir, configFile: file, jobId, requestId });
    const canResumeCancelled = prior.state === 'cancelled' && resumeCancelled;
    if (!canResumeCancelled && (!retry || !['failed', 'interrupted'].includes(prior.state))) return prior;
    attempt = (prior.attempt ?? 1) + 1;
  }
  const attemptDir = path.join(loc.dir, `attempt-${attempt}`);
  try {
    await mkdir(attemptDir);
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    return backgroundTrainingStatus({ baseDir, configFile: file, jobId, requestId });
  }
  const requestFile = path.join(attemptDir, 'request.json');
  await writeJson(requestFile, { ...request, requestId, attempt });
  const status = {
    requestId, jobId, state: 'queued', attempt, createdAt: new Date().toISOString(), device,
    logFile: path.join(attemptDir, 'worker.log'),
  };
  await writeJson(loc.statusFile, status);
  try {
    const launch = await launchBackgroundWorker(workerFile, requestFile, loc.statusFile,
      `tmf-train-${requestId}-${attempt}`, status.logFile);
    await writeJson(path.join(attemptDir, 'launch.json'), launch);
    const now = await readJsonIfExists(loc.statusFile);
    if (now?.state === 'queued' && now.attempt === attempt) await writeJson(loc.statusFile, { ...now, ...launch });
  } catch (err) {
    await writeJson(loc.statusFile, {
      ...status, state: 'failed', finishedAt: new Date().toISOString(),
      error: { code: 'WORKER_LAUNCH_FAILED', message: err.message },
    });
  }
  return backgroundTrainingStatus({ baseDir, configFile: file, jobId, requestId });
}

export async function backgroundTrainingStatus({ baseDir, configFile, jobId, requestId }) {
  requireJobId(jobId);
  if (typeof requestId !== 'string' || !/^[a-f0-9]{24}$/.test(requestId)) throw new Error('requestId must be a 24-character hex digest');
  const loc = await location({ baseDir, configFile, jobId, requestId });
  const status = await readJsonIfExists(loc.statusFile);
  if (!status) {
    if (!(await exists(loc.dir))) throw new Error('background training not found: ' + requestId);
    return { requestId, jobId, state: 'initializing', statusFile: loc.statusFile };
  }
  const launch = await readJsonIfExists(path.join(loc.dir, `attempt-${status.attempt}`, 'launch.json'));
  const withLaunch = launch ? { ...status, ...launch } : status;
  if (status.state === 'running' && status.pid) {
    try {
      process.kill(status.pid, 0);
    } catch (err) {
      if (err.code === 'ESRCH') {
        const durable = await trainingStatus({ baseDir, configFile: loc.file, jobId });
        const fresh = durable.found && durable.state?.updatedAt >= status.startedAt;
        const state = fresh && durable.state?.state === 'succeeded' ? 'succeeded'
          : fresh && durable.state?.state === 'cancelled' ? 'cancelled' : 'interrupted';
        const result = {
          ...withLaunch, state, finishedAt: new Date().toISOString(),
          jobState: fresh ? durable.state?.state : null,
        };
        if (state === 'interrupted') result.error = { code: 'WORKER_EXITED', message: 'worker exited without a matching durable job result; retry explicitly' };
        await writeJson(loc.statusFile, result);
        return { ...result, statusFile: loc.statusFile };
      }
    }
  }
  return { ...withLaunch, statusFile: loc.statusFile };
}

export async function cancelBackgroundTraining({ baseDir, configFile, jobId, requestId }) {
  const status = await backgroundTrainingStatus({ baseDir, configFile, jobId, requestId });
  if (!['queued', 'running'].includes(status.state)) return status;
  const loc = await location({ baseDir, configFile, jobId, requestId });
  await writeJson(path.join(loc.dir, `attempt-${status.attempt}`, 'cancel.json'), { requestedAt: new Date().toISOString() });
  const latest = await readJsonIfExists(status.statusFile);
  if (latest?.state === 'running' && latest.pid) {
    try { process.kill(latest.pid, 'SIGTERM'); } catch (err) { if (err.code !== 'ESRCH') throw err; }
  } else if (latest?.state === 'queued') {
    await writeJson(status.statusFile, { ...latest, state: 'cancel_requested' });
  }
  return backgroundTrainingStatus({ baseDir, configFile, jobId, requestId });
}
