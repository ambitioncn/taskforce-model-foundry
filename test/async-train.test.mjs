import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { startBackgroundTraining, backgroundTrainingStatus } from '../src/integrations/async-train.mjs';
import { synthesize } from '../src/api.mjs';

test('background training deduplicates, persists and exposes a stable receipt', async (t) => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-async-train-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  await synthesize({ baseDir: dir, taskCount: 80, pretrainCount: 80, includeNoise: false });
  const configFile = path.join(dir, 'tmf.config.json');
  await fs.writeFile(configFile, JSON.stringify({
    format: 'tmf-config-v1', project: 'async-train-fixture', runRoot: '.tmf-run',
    backend: 'builtin-js', mode: 'lora', seed: 'async-train-fixture',
    model: { maxVocab: 256, maxSeqLen: 96 },
    data: { datasets: [
      { name: 'task', path: 'data/tickets.jsonl', format: 'jsonl', license: 'synthetic' },
      { name: 'pretrain', path: 'data/corpus.jsonl', format: 'jsonl', license: 'synthetic' },
    ], config: {} },
    hyper: { steps: 8, lr: 0.005, batchSize: 4 },
    budget: { maxTokens: 200000, maxWallClockMs: 120000 },
  }));
  const jobId = 'async-train-fixture';
  const old = process.env.TMF_BACKGROUND_LAUNCHER;
  process.env.TMF_BACKGROUND_LAUNCHER = 'direct';
  t.after(() => { if (old === undefined) delete process.env.TMF_BACKGROUND_LAUNCHER; else process.env.TMF_BACKGROUND_LAUNCHER = old; });
  const first = await startBackgroundTraining({ baseDir: dir, configFile, jobId });
  assert.match(first.requestId, /^[a-f0-9]{24}$/);
  const second = await startBackgroundTraining({ baseDir: dir, configFile, jobId });
  assert.equal(second.requestId, first.requestId);
  assert.equal(second.attempt, 1);
  const oldDevice = process.env.TMF_DEVICE;
  process.env.TMF_DEVICE = first.device === 'cuda' ? 'cpu' : 'cuda';
  await assert.rejects(
    () => startBackgroundTraining({ baseDir: dir, configFile, jobId }),
    /bound to request/,
    'changed execution inputs cannot launch a second worker against one jobId',
  );
  if (oldDevice === undefined) delete process.env.TMF_DEVICE; else process.env.TMF_DEVICE = oldDevice;
  let final;
  for (let n = 0; n < 600; n++) {
    final = await backgroundTrainingStatus({ baseDir: dir, configFile, jobId, requestId: first.requestId });
    if (['succeeded', 'failed', 'interrupted'].includes(final.state)) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.equal(final.state, 'succeeded', JSON.stringify(final.error));
  assert.equal(final.jobState, 'succeeded');
  assert.ok(final.checkpoint);
  assert.ok(final.logFile);
  const resumeVariant = await startBackgroundTraining({ baseDir: dir, configFile, jobId, resumeCancelled: true });
  assert.equal(resumeVariant.requestId, first.requestId, 'resume-cancelled is an attempt option, not a second job owner');
  assert.equal(resumeVariant.attempt, 1);
});
