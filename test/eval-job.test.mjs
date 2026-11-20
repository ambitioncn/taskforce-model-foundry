import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {
  recordFingerprint, assertHoldoutIndependence, computeImprovement, assembleMetrics, EVAL_REPORT_FORMAT,
} from '../src/eval/evaluator.mjs';
import { runEvaluationJob, assertDeliverable } from '../src/jobs/evaluate.mjs';
import { buildTokenizer, encodeRecords } from '../src/train/dataset.mjs';
import builtin from '../src/train/backends/builtin-js.mjs';
import { readJson } from '../src/core/fsx.mjs';

const WORK = '.tmf-run/test-eval-job';
const MODEL_CONFIG = { dModel: 16, nLayers: 1, nHeads: 2, dFF: 32, maxSeqLen: 32 };

function rec(id, status) {
  return { prompt: `ticket: ${id}`, completion: `status=${status}` };
}

const train = [rec('t1', 'open'), rec('t2', 'closed'), rec('t3', 'open'), rec('t4', 'closed')];
const holdout = [rec('h1', 'open'), rec('h2', 'closed')];
const validation = [rec('v1', 'open')];

// -------------------------------------------------------------- independence

test('holdout fingerprints ignore nothing but are stable per record', () => {
  assert.equal(recordFingerprint(rec('h1', 'open')), recordFingerprint(rec('h1', 'open')));
  assert.notEqual(recordFingerprint(rec('h1', 'open')), recordFingerprint(rec('h1', 'closed')));
});

test('an empty holdout cannot be evaluated', () => {
  assert.throws(() => assertHoldoutIndependence({ holdout: [], train }), /non-empty holdout/);
});

test('a holdout record leaked into train is terminal, not advisory', () => {
  assert.throws(
    () => assertHoldoutIndependence({ holdout, train: [...train, holdout[0]] }),
    (err) => err.code === 'HOLDOUT_LEAKED' && /1\/2 holdout records/.test(err.message),
  );
});

test('clean splits report their sizes and a holdout fingerprint', () => {
  const summary = assertHoldoutIndependence({ holdout, train, validation });
  assert.equal(summary.ok, true);
  assert.equal(summary.holdout, 2);
  assert.equal(summary.train, 4);
  assert.match(summary.holdoutFingerprint, /^[0-9a-f]{64}$/);
});

// ------------------------------------------------------------------- deltas

test('improvement deltas are signed relative to the baseline', () => {
  const baseline = { holdout: { loss: 4, exactMatch: { accuracy: 0 } }, validation: { loss: 4 } };
  const tuned = { holdout: { loss: 1, exactMatch: { accuracy: 0.5 } }, validation: { loss: 2 } };
  const delta = computeImprovement(baseline, tuned);
  assert.equal(delta.holdoutLossRelative, 0.75);
  assert.equal(delta.holdoutLossAbsolute, 3);
  assert.equal(delta.validationLossRelative, 0.5);
  assert.equal(delta.exactMatchAbsolute, 0.5);
});

test('a regression produces a negative relative improvement', () => {
  const delta = computeImprovement(
    { holdout: { loss: 1, exactMatch: { accuracy: 0.5 } }, validation: null },
    { holdout: { loss: 2, exactMatch: { accuracy: 0.25 } }, validation: null },
  );
  assert.equal(delta.holdoutLossRelative, -1);
  assert.equal(delta.exactMatchAbsolute, -0.25);
  assert.equal(delta.validationLossRelative, null);
});

test('metrics without a tuned model carry no improvement block', () => {
  const metrics = assembleMetrics({ baseline: { holdout: { loss: 1 } }, tuned: null });
  assert.equal(metrics.improvement, null);
});

// --------------------------------------------------------------- the job

test('the evaluation job scores checkpoints from disk and gates both ways', async (t) => {
  await fs.rm(WORK, { recursive: true, force: true });
  const tokenizer = buildTokenizer([...train, ...holdout, ...validation], { maxVocab: 64 });
  const encoded = {
    holdout: encodeRecords(holdout, tokenizer, { maxSeqLen: 32 }).examples,
    validation: encodeRecords(validation, tokenizer, { maxSeqLen: 32 }).examples,
  };
  const trainExamples = encodeRecords(train, tokenizer, { maxSeqLen: 32 }).examples;

  const handle = await builtin.createModel({ modelConfig: MODEL_CONFIG, tokenizer, seed: 3 });
  const baseFile = `${WORK}/base.json`;
  await builtin.saveModel({ handle, file: baseFile, kind: 'base', meta: { stage: 'test-base' } });
  await builtin.train({
    handle,
    examples: trainExamples,
    mode: 'lora',
    hyper: { lr: 0.02, batchSize: 4, lora: { rank: 4, alpha: 8, targets: ['attn.q', 'attn.v'] } },
    budget: { maxSteps: 40, maxSeconds: 120 },
    seed: 'test-lora',
  });
  const adapterFile = `${WORK}/adapter.json`;
  await builtin.saveModel({
    handle,
    file: adapterFile,
    kind: 'adapter',
    meta: { stage: 'test-lora', baseCheckpoint: baseFile },
  });

  const lenient = [
    { id: 'holdout_loss_improvement', metric: 'improvement.holdoutLossRelative', comparator: 'gte', threshold: -5 },
    { id: 'holdout_loss_absolute', metric: 'tuned.holdout.loss', comparator: 'lte', threshold: 100 },
  ];
  const reportFile = `${WORK}/eval-report.json`;
  const passing = await runEvaluationJob({
    backend: builtin,
    baseCheckpoint: baseFile,
    adapterCheckpoint: adapterFile,
    encoded,
    records: { holdout, validation, train },
    gates: lenient,
    lineage: { dataReport: 'synthetic-inline', trainedBy: 'eval-job.test' },
    reportFile,
    jobId: 'test-eval-pass',
  });

  assert.equal(passing.format, EVAL_REPORT_FORMAT);
  assert.equal(passing.verdict.status, 'passed');
  assert.equal(passing.verdict.improvementClaimAllowed, true);
  assert.equal(passing.verdict.deliveryEligible, true);
  assert.equal(assertDeliverable(passing), passing);

  // Baseline and tuned are both measured, and the deltas connect them.
  assert.equal(typeof passing.metrics.baseline.holdout.loss, 'number');
  assert.equal(typeof passing.metrics.tuned.holdout.loss, 'number');
  assert.equal(typeof passing.metrics.improvement.holdoutLossRelative, 'number');
  assert.equal(passing.metrics.baseline.holdout.exactMatch.total, 2);

  // Lineage binds the report to the exact checkpoint bytes and gate set.
  assert.match(passing.inputs.baseCheckpoint.sha256, /^[0-9a-f]{64}$/);
  assert.equal(passing.inputs.adapterCheckpoint.file, adapterFile);
  assert.equal(passing.inputs.splits.trainWithheld, 4);
  assert.equal(passing.lineage.dataReport, 'synthetic-inline');
  assert.equal(passing.lineage.gateDefinitions.length, 2);
  assert.equal(passing.lineage.baseFingerprint, passing.lineage.tunedFingerprint);
  assert.equal(passing.lineage.adapterBoundToBase, true);
  assert.match(passing.reportDigest, /^[0-9a-f]{64}$/);

  const onDisk = await readJson(reportFile);
  assert.equal(onDisk.jobId, 'test-eval-pass');
  assert.equal(onDisk.verdict.status, 'passed');

  await t.test('an unreachable threshold fails the gate and blocks delivery', async () => {
    const failing = await runEvaluationJob({
      backend: builtin,
      baseCheckpoint: baseFile,
      adapterCheckpoint: adapterFile,
      encoded,
      records: { holdout, validation, train },
      gates: [
        { id: 'impossible_exact_match', metric: 'tuned.holdout.exactMatch.accuracy', comparator: 'gte', threshold: 1 },
        ...lenient,
      ],
      jobId: 'test-eval-fail',
    });
    assert.equal(failing.verdict.status, 'failed');
    assert.equal(failing.verdict.improvementClaimAllowed, false);
    assert.equal(failing.verdict.deliveryEligible, false);
    assert.equal(failing.gates.failed[0].id, 'impossible_exact_match');
    assert.throws(() => assertDeliverable(failing), (err) => err.code === 'EVAL_NOT_DELIVERABLE');
  });

  await t.test('a baseline-only run may not claim improvement', async () => {
    const baselineOnly = await runEvaluationJob({
      backend: builtin,
      baseCheckpoint: baseFile,
      encoded,
      records: { holdout, validation, train },
      gates: lenient,
      jobId: 'test-eval-baseline',
    });
    assert.equal(baselineOnly.verdict.status, 'baseline_only');
    assert.equal(baselineOnly.metrics.tuned, null);
    assert.throws(() => assertDeliverable(baselineOnly), /not deliverable/);
  });

  await t.test('a missing checkpoint is reported by path', async () => {
    await assert.rejects(
      () => runEvaluationJob({
        backend: builtin,
        baseCheckpoint: `${WORK}/nope.json`,
        encoded,
        records: { holdout, train },
      }),
      /checkpoint not found/,
    );
  });

  await t.test('the job refuses to run on a leaked holdout', async () => {
    await assert.rejects(
      () => runEvaluationJob({
        backend: builtin,
        baseCheckpoint: baseFile,
        adapterCheckpoint: adapterFile,
        encoded,
        records: { holdout, train: [...train, holdout[0]] },
        gates: lenient,
      }),
      (err) => err.code === 'HOLDOUT_LEAKED',
    );
  });
});

test('recordFingerprint separates text-only records instead of collapsing them', () => {
  // Regression: keying on prompt/completion alone made every text-only record
  // hash "undefined|undefined", so assertHoldoutIndependence reported a clean
  // holdout as fully leaked.
  const a = { text: 'the finance team reviewed the billing refund' };
  const b = { text: 'the people team reviewed the hiring escalation' };
  assert.notEqual(recordFingerprint(a), recordFingerprint(b));
  assert.equal(recordFingerprint(a), recordFingerprint({ text: a.text }));
  assert.notEqual(
    recordFingerprint({ prompt: 'route', completion: 'queue=security' }),
    recordFingerprint({ prompt: 'route', completion: 'queue=billing' }),
  );
  assert.throws(() => recordFingerprint({ id: 'x' }), /prompt\/completion or text/);
  const report = assertHoldoutIndependence({ holdout: [a], train: [b], validation: [] });
  assert.equal(report.ok, true);
  assert.equal(report.holdout, 1);
});

// ------------------------------------------- full-parameter (no adapter) path

// Regression: evaluate only computed a tuned side when an *adapter* was
// supplied, so pretrain / full-finetune jobs could only ever report
// baseline_only -- no gates, and export blocked forever.
test('a full-parameter tuned checkpoint is gated like an adapter run', async (t) => {
  const work = `${WORK}-full`;
  await fs.rm(work, { recursive: true, force: true });
  const tokenizer = buildTokenizer([...train, ...holdout, ...validation], { maxVocab: 64 });
  const encoded = {
    holdout: encodeRecords(holdout, tokenizer, { maxSeqLen: 32 }).examples,
    validation: encodeRecords(validation, tokenizer, { maxSeqLen: 32 }).examples,
  };
  const trainExamples = encodeRecords(train, tokenizer, { maxSeqLen: 32 }).examples;

  const handle = await builtin.createModel({ modelConfig: MODEL_CONFIG, tokenizer, seed: 5 });
  const baseFile = `${work}/base.json`;
  await builtin.saveModel({ handle, file: baseFile, kind: 'base', meta: { stage: 'full-base' } });

  const result = await builtin.train({
    handle,
    examples: trainExamples,
    mode: 'full-finetune',
    hyper: { lr: 0.02, batchSize: 4 },
    budget: { maxSteps: 40, maxSeconds: 120 },
    seed: 'test-full',
  });
  assert.equal(result.metrics.mode, 'full-finetune');
  // The whole point of the mode: far more capacity than a rank-4 adapter.
  assert.ok(result.metrics.trainableParams > 1000, `trainableParams=${result.metrics.trainableParams}`);

  const tunedFile = `${work}/tuned.json`;
  await builtin.saveModel({ handle, file: tunedFile, kind: 'base', meta: { stage: 'full-tuned' } });

  const gates = [
    { id: 'holdout_loss_improvement', metric: 'improvement.holdoutLossRelative', comparator: 'gte', threshold: -5 },
  ];
  const report = await runEvaluationJob({
    backend: builtin,
    baseCheckpoint: baseFile,
    tunedCheckpoint: tunedFile,
    encoded,
    records: { holdout, validation, train },
    gates,
    reportFile: `${work}/eval-report.json`,
    jobId: 'test-eval-full',
  });

  assert.equal(report.verdict.status, 'passed');
  assert.equal(report.verdict.deliveryEligible, true);
  assert.equal(typeof report.metrics.tuned.holdout.loss, 'number');
  assert.equal(typeof report.metrics.improvement.holdoutLossRelative, 'number');
  assert.equal(report.gates.results.length, 1);

  // The tuned bytes are pinned, and no adapter is claimed.
  assert.equal(report.inputs.tunedCheckpoint.file, tunedFile);
  assert.match(report.inputs.tunedCheckpoint.sha256, /^[0-9a-f]{64}$/);
  assert.equal(report.inputs.adapterCheckpoint, null);
  assert.equal(report.lineage.adapterBoundToBase, null);
  // Full-parameter training moves the base weights, so the fingerprints differ.
  assert.notEqual(report.lineage.tunedFingerprint, report.lineage.baseFingerprint);

  await t.test('a tuned checkpoint of an unknown kind is terminal, not silently random', async () => {
    const bogus = `${work}/bogus.json`;
    const payload = await readJson(tunedFile);
    await fs.writeFile(bogus, JSON.stringify({ ...payload, kind: 'tuned' }));
    await assert.rejects(
      () => builtin.loadModel({ file: bogus }),
      /unsupported checkpoint kind "tuned"/,
    );
  });

  await fs.rm(work, { recursive: true, force: true });
});
