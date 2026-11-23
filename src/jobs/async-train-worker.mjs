import path from 'node:path';
import { readJson, writeJson, exists } from '../core/fsx.mjs';
import { sha256File } from '../core/hash.mjs';
import { train, status as trainingStatus } from '../api.mjs';

const [, , requestFile, statusFile] = process.argv;
if (!requestFile || !statusFile) throw new Error('training worker requires request and status files');
const request = await readJson(requestFile);
const cancelFile = path.join(path.dirname(requestFile), 'cancel.json');
const old = await readJson(statusFile);
if (old.requestId !== request.requestId || old.attempt !== request.attempt) {
  throw new Error('background training attempt no longer owns the status file');
}
if (await sha256File(request.configFile) !== request.configDigest) {
  throw new Error('background training config changed since submission');
}
if (old.state === 'cancel_requested' || await exists(cancelFile)) {
  await writeJson(statusFile, { ...old, state: 'cancelled', finishedAt: new Date().toISOString() });
  process.exit(130);
}
process.env.TMF_DEVICE = request.device;
process.env.TMF_PYTHON = request.pythonBin;
if (request.hfMaxMemoryMb !== null) process.env.TMF_HF_MAX_MEMORY_MB = request.hfMaxMemoryMb;
process.chdir(path.dirname(statusFile));
const controller = new AbortController();
for (const signal of ['SIGTERM', 'SIGINT']) {
  process.on(signal, () => controller.abort(new Error(`operator cancelled with ${signal}`)));
}
const common = { ...old };
await writeJson(statusFile, { ...common, state: 'running', pid: process.pid, startedAt: new Date().toISOString() });
if (await exists(cancelFile)) controller.abort(new Error('operator cancelled before training started'));
try {
  const result = await train({
    baseDir: path.dirname(request.configFile), configFile: request.configFile, jobId: request.jobId,
    resume: request.resume, resumeCancelled: request.resumeCancelled, auditEveryStep: request.auditEveryStep, signal: controller.signal,
  });
  await writeJson(statusFile, {
    ...common, state: result.job.state, pid: process.pid, finishedAt: new Date().toISOString(),
    jobState: result.job.state, checkpoint: result.artifacts?.adapterCheckpoint ?? result.artifacts?.tunedCheckpoint ?? null,
  });
  if (result.job.state !== 'succeeded') process.exitCode = 1;
} catch (err) {
  const durable = await trainingStatus({ baseDir: path.dirname(request.configFile), configFile: request.configFile, jobId: request.jobId });
  const cancelled = controller.signal.aborted || durable.state?.state === 'cancelled';
  await writeJson(statusFile, {
    ...common, state: cancelled ? 'cancelled' : 'failed', pid: process.pid,
    finishedAt: new Date().toISOString(), jobState: durable.state?.state ?? null,
    error: { code: err?.code ?? 'action-failed', message: err?.message ?? String(err) },
  });
  process.exitCode = cancelled ? 130 : 1;
}
