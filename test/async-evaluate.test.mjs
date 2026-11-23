import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { startBackgroundEvaluation, backgroundEvaluationStatus } from '../src/jobs/async-evaluate.mjs';
import { synthesize, train } from '../src/api.mjs';

const jobId = 'async-eval-fixture';

async function project() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-async-'));
  await synthesize({ baseDir: dir, taskCount: 80, pretrainCount: 80, includeNoise: false });
  const configFile = path.join(dir, 'tmf.config.json');
  await fs.writeFile(configFile, JSON.stringify({
    format: 'tmf-config-v1', project: 'async-eval-fixture', runRoot: '.tmf-run',
    backend: 'builtin-js', mode: 'lora', seed: 'async-eval-fixture',
    model: { maxVocab: 256, maxSeqLen: 96 },
    data: { datasets: [
      { name: 'task', path: 'data/tickets.jsonl', format: 'jsonl', license: 'synthetic', source: 'tmf synth' },
      { name: 'pretrain', path: 'data/corpus.jsonl', format: 'jsonl', license: 'synthetic', source: 'tmf synth' },
    ], config: {} },
    hyper: { steps: 8, lr: 0.005, batchSize: 4 },
    budget: { maxTokens: 200000, maxWallClockMs: 120000 },
  }));
  const result = await train({ baseDir: dir, configFile, jobId });
  assert.equal(result.job.state, 'succeeded');
  return { dir, configFile, reportFile: path.join(dir, 'background-report.json') };
}

async function settled(args, requestId) {
  for (let n = 0; n < 900; n++) {
    const status = await backgroundEvaluationStatus({ ...args, jobId, requestId });
    if (['succeeded', 'failed', 'interrupted'].includes(status.state)) return status;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('background evaluation did not settle within 90 seconds');
}

test('background output belongs to its request and a stale report cannot prove success', async (t) => {
  const args = await project();
  t.after(() => fs.rm(args.dir, { recursive: true, force: true }));
  await fs.writeFile(args.reportFile, '{"verdict":{"status":"passed"},"old":true}\n');
  const first = await startBackgroundEvaluation({ ...args, jobId });
  assert.match(first.requestId, /^[a-f0-9]{24}$/);
  assert.equal((await startBackgroundEvaluation({ ...args, jobId })).requestId, first.requestId);
  const final = await settled(args, first.requestId);
  assert.equal(final.state, 'succeeded', JSON.stringify(final.error));
  const report = JSON.parse(await fs.readFile(args.reportFile, 'utf8'));
  assert.equal(report.backgroundRequest.requestId, first.requestId);
  assert.equal(report.backgroundRequest.attempt, 1);
  assert.equal(final.reportDigest, report.reportDigest);
  assert.equal(report.execution.device, null);

  const statusFile = path.join(args.dir, '.tmf-run', jobId, 'plugin-evaluations', first.requestId, 'status.json');
  await fs.writeFile(statusFile, JSON.stringify({ ...final, state: 'running', attempt: 2, pid: 999999999, reportFile: args.reportFile }));
  const stale = await backgroundEvaluationStatus({ ...args, jobId, requestId: first.requestId });
  assert.equal(stale.state, 'interrupted');
  const retried = await startBackgroundEvaluation({ ...args, jobId, retry: true });
  assert.equal(retried.requestId, first.requestId);
  assert.equal((await settled(args, first.requestId)).state, 'succeeded');
  assert.equal(JSON.parse(await fs.readFile(args.reportFile, 'utf8')).backgroundRequest.attempt, 3);
});
