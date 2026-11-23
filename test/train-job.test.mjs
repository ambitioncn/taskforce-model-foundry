import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { runTrainingJob, trainingArtifacts, TRAIN_JOB_KIND } from '../src/jobs/train.mjs';
import { readAuditLog, readJobState } from '../src/jobs/runner.mjs';
import { writeSyntheticCorpus } from '../src/data/synth.mjs';
import builtin from '../src/train/backends/builtin-js.mjs';
import { readJson } from '../src/core/fsx.mjs';

const WORK = '.tmf-run/test-train-job';
const MODEL_CONFIG = { dModel: 32, nLayers: 2, nHeads: 2, dFF: 64, maxSeqLen: 64 };

let corpus = null;

async function fixture() {
  if (!corpus) {
    await fs.rm(WORK, { recursive: true, force: true });
    corpus = await writeSyntheticCorpus(path.join(WORK, 'corpus'), {
      taskCount: 60, pretrainCount: 20, seed: 'train-job-test',
    });
  }
  return {
    root: WORK,
    backend: builtin,
    datasets: corpus.datasets,

    modelConfig: MODEL_CONFIG,
    hyper: { batchSize: 4, lr: 0.01 },
    maxSeqLen: 64,
  };
}

test('training job runs data -> base -> train as one auditable job', async () => {
  const base = await fixture();
  const job = await runTrainingJob({ ...base, jobId: 'happy', trainBudget: { maxSteps: 6 } });

  assert.equal(job.state, 'succeeded');
  assert.equal(job.kind, TRAIN_JOB_KIND);
  assert.deepEqual(job.completedSteps.map((s) => s.name), ['data', 'base', 'train', 'finalize']);

  // Real post-training: the weights actually moved and the metrics are measured,
  // not asserted by the job wrapper.
  assert.ok(job.result.train.metrics.parameterL2Delta > 0);
  assert.equal(job.result.train.metrics.steps, 6);
  assert.equal(job.result.train.kind, 'adapter');

  const artifacts = trainingArtifacts(job);
  assert.ok(await fs.stat(artifacts.baseCheckpoint));
  assert.ok(await fs.stat(artifacts.adapterCheckpoint));
  assert.equal(artifacts.tunedCheckpoint, null);
  const manifest = await readJson(job.result.manifestFile);
  assert.equal(manifest.jobId, 'happy');
  assert.equal(manifest.base.sha256, job.result.base.sha256);

  // The adapter is bound to the base it was trained against.
  const adapter = await readJson(artifacts.adapterCheckpoint);
  assert.equal(adapter.kind, 'adapter');
  assert.equal(adapter.baseFingerprint, job.result.train.baseFingerprint);

  const audit = await readAuditLog(WORK, 'happy');
  const events = audit.map((e) => e.event);
  for (const expected of ['job.started', 'data.prepared', 'base.saved', 'train.started', 'train.finished', 'train.manifest', 'job.succeeded']) {
    assert.ok(events.includes(expected), `audit log is missing ${expected}`);
  }
  // Per-step training telemetry is in the log, so a reviewer can see the loss
  // curve of the run that produced the checkpoint.
  const steps = audit.filter((e) => e.event === 'train.step');
  assert.equal(steps.length, 6);
  assert.ok(steps[0].loss > 0);
});

test('replaying a finished training job returns the stored result without retraining', async () => {
  const base = await fixture();
  const first = await runTrainingJob({ ...base, jobId: 'replay', trainBudget: { maxSteps: 4 } });
  const before = (await readAuditLog(WORK, 'replay')).length;
  const again = await runTrainingJob({ ...base, jobId: 'replay', trainBudget: { maxSteps: 4 } });

  assert.equal(again.replayed, true);
  assert.equal(again.state, 'succeeded');
  assert.equal(again.result.train.sha256, first.result.train.sha256);
  assert.equal((await readAuditLog(WORK, 'replay')).length, before, 'a replay must not append new audit events');
});

test('changing the config under an existing jobId is refused', async () => {
  const base = await fixture();
  await runTrainingJob({ ...base, jobId: 'digest', trainBudget: { maxSteps: 3 } });
  await assert.rejects(
    () => runTrainingJob({ ...base, jobId: 'digest', trainBudget: { maxSteps: 9 } }),
    /different input\/plan digest/,
  );
});

test('external quality contract is frozen in training input and manifest', async () => {
  const base = await fixture();
  const externalQuality = { suiteSha256: 'a'.repeat(64), scorerSha256: 'b'.repeat(64), gates: [
    { id: 'quality', metric: 'quality.score', comparator: 'gte', threshold: 0.7 },
  ] };
  const job = await runTrainingJob({ ...base, jobId: 'quality-frozen', trainBudget: { maxSteps: 1 }, externalQuality });
  assert.equal(job.state, 'succeeded');
  assert.deepEqual(trainingArtifacts(job).externalQuality, externalQuality);
  assert.deepEqual((await readJson(job.result.manifestFile)).externalQuality, externalQuality);
  await assert.rejects(
    () => runTrainingJob({ ...base, jobId: 'quality-frozen', trainBudget: { maxSteps: 1 }, externalQuality: { ...externalQuality, suiteSha256: 'c'.repeat(64) } }),
    /different input\/plan digest/,
  );
});

test('cancellation stops at a step boundary and the job resumes without redoing data or base', async () => {
  const base = await fixture();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort('operator cancelled'), 200);
  await assert.rejects(
    () => runTrainingJob({
      ...base, jobId: 'cancel', trainBudget: { maxSteps: 400 }, signal: controller.signal, auditEveryStep: 50,
    }),
    (error) => error.code === 'JOB_CANCELLED',
  );
  clearTimeout(timer);

  const cancelled = await readJobState(WORK, 'cancel');
  assert.equal(cancelled.state, 'cancelled');
  assert.deepEqual(cancelled.completedSteps.map((s) => s.name), ['data', 'base']);
  const baseFile = cancelled.completedSteps.find((s) => s.name === 'base').output.file;

  // A cancelled job is paused, not void: resuming continues from the first
  // step that never completed and reuses the base checkpoint on disk.
  const resumed = await runTrainingJob({
    ...base, jobId: 'cancel', trainBudget: { maxSteps: 400 }, resumeCancelled: true, auditEveryStep: 200,
  });
  assert.equal(resumed.state, 'succeeded');
  assert.equal(resumed.result.base.file, baseFile);
  assert.equal(resumed.result.train.metrics.steps, 400);
  assert.ok(resumed.result.train.metrics.parameterL2Delta > 0);

  const audit = await readAuditLog(WORK, 'cancel');
  assert.deepEqual(audit.filter((e) => e.event === 'step.skipped').map((e) => e.step), ['data', 'base']);
  assert.equal(audit.filter((e) => e.event === 'job.cancelled').length, 1);
  assert.equal(audit.filter((e) => e.event === 'job.resumed').length, 1);
});

test('a cancelled job is not resumed by accident', async () => {
  const base = await fixture();
  const controller = new AbortController();
  controller.abort('before start');
  await assert.rejects(
    () => runTrainingJob({ ...base, jobId: 'cancel-strict', trainBudget: { maxSteps: 3 }, signal: controller.signal }),
    (error) => error.code === 'JOB_CANCELLED',
  );
  // Without resumeCancelled the re-run is an idempotent replay, not a restart.
  const replay = await runTrainingJob({ ...base, jobId: 'cancel-strict', trainBudget: { maxSteps: 3 } });
  assert.equal(replay.replayed, true);
  assert.equal(replay.state, 'cancelled');
  assert.equal(replay.result, null);
});

test('a token budget trips inside the training loop and stops the job', async () => {
  const base = await fixture();
  await assert.rejects(
    () => runTrainingJob({
      ...base, jobId: 'budget', trainBudget: { maxSteps: 400 }, budget: { maxTokens: 300 }, auditEveryStep: 50,
    }),
    (error) => error.code === 'JOB_BUDGET_EXCEEDED' && error.details.limit === 'maxTokens',
  );
  const state = await readJobState(WORK, 'budget');
  assert.equal(state.state, 'budget_exceeded');
  assert.ok(state.budget.usage.tokens <= 300, 'usage must never exceed the cap it reports');
  // The cap stops the run early: the training step never reaches max_steps.
  const history = await fs.stat(path.join(WORK, 'budget', 'model', 'adapter.json')).catch(() => null);
  assert.equal(history, null, 'an over-budget job must not leave a tuned checkpoint behind');
});

test('an unsupported mode is rejected before any work happens', async () => {
  const base = await fixture();
  await assert.rejects(
    () => runTrainingJob({ ...base, jobId: 'bad-mode', mode: 'rlhf' }),
    /does not support mode "rlhf"/,
  );
  assert.equal(await fs.stat(path.join(WORK, 'bad-mode')).catch(() => null), null);
});

test.after(async () => {
  await fs.rm(WORK, { recursive: true, force: true });
});
