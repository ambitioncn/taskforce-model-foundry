// Background evaluations use a transient user service when available, outside
// the Gateway cgroup. Every attempt owns its report; old output is never proof
// that a new attempt finished.
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { open, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadConfig } from '../config.mjs';
import { ensureDir, exists, readJsonIfExists, writeJson } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';

const workerFile = fileURLToPath(new URL('./async-evaluate-worker.mjs', import.meta.url));
const execFileAsync = promisify(execFile);

export function requireJobId(jobId) {
  if (typeof jobId !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(jobId) || jobId === '..') {
    throw new Error('background work requires a safe, explicit jobId');
  }
  return jobId;
}

async function location({ baseDir, configFile, jobId, requestId }) {
  const { file, config } = await loadConfig({ file: configFile, baseDir });
  const root = path.join(config.runRootAbsolute, requireJobId(jobId), 'plugin-evaluations');
  const dir = path.join(root, requestId);
  return { file, root, dir, statusFile: path.join(dir, 'status.json') };
}

function reportMatches(report, status) {
  if (report?.backgroundRequest?.requestId !== status.requestId || report?.backgroundRequest?.attempt !== status.attempt) return false;
  return report.reportDigest === sha256Json({ ...report, reportDigest: undefined });
}

export async function launchBackgroundWorker(workerPath, requestFile, statusFile, unit, logFile) {
  const launcher = process.env.TMF_BACKGROUND_LAUNCHER ?? process.env.TMF_EVAL_LAUNCHER;
  if (launcher !== 'direct' && process.platform === 'linux') {
    try {
      await execFileAsync('systemd-run', ['--user', '--collect', '--unit=' + unit, process.execPath, workerPath, requestFile, statusFile], { timeout: 10000 });
      return { launcher: 'systemd', unit, restartDurable: true };
    } catch (err) {
      if (launcher === 'systemd') throw err;
    }
  }
  const log = await open(logFile, 'a');
  try {
    const child = spawn(process.execPath, [workerPath, requestFile, statusFile], {
      detached: true, stdio: ['ignore', log.fd, log.fd], env: process.env,
    });
    if (!child.pid) throw new Error('background evaluation worker did not start');
    child.unref();
    return { launcher: 'direct', restartDurable: false, pid: child.pid };
  } finally {
    await log.close();
  }
}

export async function startBackgroundEvaluation({ baseDir, configFile, jobId, gates = null, reportFile = null, maxNewTokens = 48, retry = false }) {
  requireJobId(jobId);
  const { file, config } = await loadConfig({ file: configFile, baseDir });
  const output = path.resolve(config.baseDir, reportFile ?? path.join(config.runRootAbsolute, jobId, 'eval-report.json'));
  const device = process.env.TMF_DEVICE ?? 'cpu';
  if (!['cpu', 'cuda'].includes(device)) throw new Error('unsupported TMF_DEVICE=' + device);
  const request = {
    configFile: file, configDigest: await sha256File(file), jobId, gates, reportFile: output,
    maxNewTokens, device, pythonBin: process.env.TMF_PYTHON ?? 'python3',
    hfMaxMemoryMb: process.env.TMF_HF_MAX_MEMORY_MB ?? null,
  };
  const requestId = sha256Json(request).slice(0, 24);
  const loc = await location({ baseDir, configFile: file, jobId, requestId });
  await ensureDir(loc.root);
  let attempt = 1;
  try {
    await mkdir(loc.dir);
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    const prior = await backgroundEvaluationStatus({ baseDir, configFile: file, jobId, requestId });
    if (!retry || !['interrupted', 'failed'].includes(prior.state)) return prior;
    attempt = (prior.attempt ?? 1) + 1;
  }
  const attemptDir = path.join(loc.dir, 'attempt-' + attempt);
  try {
    await mkdir(attemptDir);
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    return backgroundEvaluationStatus({ baseDir, configFile: file, jobId, requestId });
  }
  const requestFile = path.join(attemptDir, 'request.json');
  await writeJson(requestFile, { ...request, requestId, attempt, stageReportFile: path.join(attemptDir, 'report.json') });
  const status = { requestId, jobId, state: 'queued', attempt, reportFile: output, createdAt: new Date().toISOString(), device };
  await writeJson(loc.statusFile, status);
  try {
    const launch = await launchBackgroundWorker(workerFile, requestFile, loc.statusFile, 'tmf-eval-' + requestId + '-' + attempt, path.join(attemptDir, 'worker.log'));
    const now = await readJsonIfExists(loc.statusFile);
    if (now?.state === 'queued' && now.attempt === attempt) await writeJson(loc.statusFile, { ...now, ...launch });
  } catch (err) {
    await writeJson(loc.statusFile, { ...status, state: 'failed', error: { code: 'WORKER_LAUNCH_FAILED', message: err.message }, finishedAt: new Date().toISOString() });
  }
  return backgroundEvaluationStatus({ baseDir, configFile: file, jobId, requestId });
}

export async function backgroundEvaluationStatus({ baseDir, configFile, jobId, requestId }) {
  requireJobId(jobId);
  if (typeof requestId !== 'string' || !/^[a-f0-9]{24}$/.test(requestId)) throw new Error('requestId must be a 24-character hex digest');
  const loc = await location({ baseDir, configFile, jobId, requestId });
  const status = await readJsonIfExists(loc.statusFile);
  if (!status) {
    if (!(await exists(loc.dir))) throw new Error('background evaluation not found: ' + requestId);
    return { requestId, jobId, state: 'initializing', statusFile: loc.statusFile };
  }
  if (status.state === 'running' && status.pid) {
    try {
      process.kill(status.pid, 0);
    } catch (err) {
      if (err.code === 'ESRCH') {
        const report = await readJsonIfExists(status.reportFile);
        if (reportMatches(report, status)) {
          const succeeded = { ...status, state: 'succeeded', finishedAt: report.finishedAt, reportDigest: report.reportDigest, verdict: report.verdict, gates: report.gates };
          delete succeeded.error;
          await writeJson(loc.statusFile, succeeded);
          return { ...succeeded, statusFile: loc.statusFile };
        }
        const interrupted = { ...status, state: 'interrupted', finishedAt: new Date().toISOString(), error: { code: 'WORKER_EXITED', message: 'worker exited without a matching final report; retry the same request explicitly' } };
        await writeJson(loc.statusFile, interrupted);
        return { ...interrupted, statusFile: loc.statusFile };
      }
    }
  }
  return { ...status, statusFile: loc.statusFile };
}
