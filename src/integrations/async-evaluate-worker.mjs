import { readJson, writeJson } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { evaluate } from '../api.mjs';
import path from 'node:path';

const [, , requestFile, statusFile] = process.argv;
if (!requestFile || !statusFile) throw new Error('worker requires request and status files');
const request = await readJson(requestFile);
const old = await readJson(statusFile);
if (old.requestId !== request.requestId || old.attempt !== request.attempt) {
  throw new Error('background request attempt no longer owns the status file');
}
if (await sha256File(request.configFile) !== request.configDigest) {
  throw new Error('background request config changed since submission');
}
process.env.TMF_DEVICE = request.device;
process.env.TMF_PYTHON = request.pythonBin;
// The HF backend uses cwd for its transient spec/output directories. Keep
// separate background requests from racing on those files.
process.chdir(path.dirname(statusFile));
const common = { ...old };
await writeJson(statusFile, { ...common, state: 'running', pid: process.pid, startedAt: new Date().toISOString() });
try {
  const { report } = await evaluate({ ...request, reportFile: request.stageReportFile });
  const published = { ...report, backgroundRequest: { requestId: request.requestId, attempt: request.attempt } };
  delete published.reportFile;
  published.reportDigest = sha256Json({ ...published, reportDigest: undefined });
  await writeJson(request.reportFile, published);
  await writeJson(statusFile, {
    ...common, state: 'succeeded', pid: process.pid, finishedAt: new Date().toISOString(),
    reportDigest: published.reportDigest, verdict: published.verdict, gates: published.gates,
  });
} catch (err) {
  await writeJson(statusFile, {
    ...common, state: 'failed', pid: process.pid, finishedAt: new Date().toISOString(),
    error: { code: err?.code ?? 'action-failed', message: err?.message ?? String(err) },
  });
  process.exitCode = 1;
}
