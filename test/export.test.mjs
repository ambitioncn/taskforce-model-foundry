import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp, readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';

import { exportModel, assertDeliverable, ExportBlockedError, EXPORT_FORMAT } from '../src/export/package.mjs';

async function fixture() {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'tmf-export-'));
  await mkdir(path.join(dir, 'run'), { recursive: true });
  const base = path.join(dir, 'run', 'base.json');
  const adapter = path.join(dir, 'run', 'adapter.json');
  const dataReport = path.join(dir, 'run', 'data-report.json');
  await writeFile(base, JSON.stringify({ weights: [1, 2, 3] }));
  await writeFile(adapter, JSON.stringify({ lora: [0.5] }));
  await writeFile(dataReport, JSON.stringify({ records: 128, dedupeRemoved: 3, pii: { findings: 0 } }));
  return {
    dir,
    artifacts: {
      jobId: 'train-demo',
      dataDir: path.join(dir, 'run', 'data'),
      dataReportFile: dataReport,
      baseCheckpoint: base,
      adapterCheckpoint: adapter,
      tunedCheckpoint: null,
      metrics: { steps: 40, finalLoss: 0.42, parameterL2Delta: 19.11 },
    },
  };
}

function passedReport(overrides = {}) {
  return {
    jobId: 'eval-demo',
    backend: { id: 'builtin-js', label: 'builtin' },
    inputs: { splits: { holdout: 24, validation: 24, trainWithheld: 80 } },
    independence: { verdict: 'independent' },
    metrics: {
      baseline: { holdout: { loss: 2.4, exactMatch: { accuracy: 0.04 } } },
      tuned: { holdout: { loss: 0.9, exactMatch: { accuracy: 0.5 } } },
      improvement: { holdoutLossRelative: 0.625, exactMatchAbsolute: 0.46, validationLossRelative: 0.6 },
    },
    gates: {
      passed: true,
      failed: [],
      summary: '5/5 gates passed',
      results: [{ id: 'holdout_loss_improvement', metric: 'improvement.holdoutLossRelative', comparator: '>=', threshold: 0.3, value: 0.625, passed: true }],
    },
    verdict: { status: 'passed', improvementClaimAllowed: true, deliveryEligible: true, summary: 'improvement confirmed' },
    lineage: { tokenizerFingerprint: 'abc123' },
    reportDigest: 'deadbeef',
    ...overrides,
  };
}

test('assertDeliverable only admits a passed, gated evaluation', () => {
  assert.equal(assertDeliverable(passedReport()).verdict.status, 'passed');
  for (const bad of [
    null,
    undefined,
    {},
    passedReport({ verdict: { status: 'failed', deliveryEligible: false }, gates: { passed: false, failed: [{ id: 'exact_match' }], results: [] } }),
    passedReport({ verdict: { status: 'baseline_only', deliveryEligible: false } }),
    // verdict and gates disagree: still refused
    passedReport({ gates: { passed: false, failed: [], results: [] } }),
    passedReport({ verdict: { status: 'passed', deliveryEligible: false } }),
  ]) {
    assert.throws(() => assertDeliverable(bad), ExportBlockedError);
  }
});

test('exportModel writes a checksummed, lineage-bearing bundle', async () => {
  const { dir, artifacts } = await fixture();
  const out = path.join(dir, 'export');
  const result = await exportModel({
    outDir: out,
    artifacts,
    evalReport: passedReport(),
    config: { project: 'demo', mode: 'adapter', seed: 7, backend: 'builtin-js', model: { dim: 48 }, hyper: { lr: 0.01 }, trainBudget: { maxSteps: 40 }, data: { datasets: ['synthetic'] } },
    version: '1.2.3',
    toolkitVersion: '0.1.0',
    clock: () => '2026-11-18T00:00:00.000Z',
  });

  const manifest = JSON.parse(await readFile(result.manifestFile, 'utf8'));
  assert.equal(manifest.format, EXPORT_FORMAT);
  assert.equal(manifest.version, '1.2.3');
  assert.equal(manifest.kind, 'adapter');
  assert.equal(manifest.toolkit.version, '0.1.0');
  assert.equal(manifest.createdAt, '2026-11-18T00:00:00.000Z');

  // data / parameter / evaluation lineage are all present and real
  assert.deepEqual(manifest.lineage.parameters.hyper, { lr: 0.01 });
  assert.equal(manifest.lineage.parameters.seed, 7);
  assert.equal(manifest.lineage.data.synthetic, true);
  assert.deepEqual(manifest.lineage.data.splits, { holdout: 24, validation: 24, trainWithheld: 80 });
  assert.equal(manifest.lineage.training.jobId, 'train-demo');
  assert.equal(manifest.lineage.evaluation.reportDigest, 'deadbeef');
  assert.equal(manifest.lineage.evaluation.gates.length, 1);
  assert.equal(manifest.metrics.holdoutLoss, 0.9);

  // every declared file exists and its recorded digest is the real one
  const names = (await readdir(out)).sort();
  assert.deepEqual(names, ['checksums.txt', 'data-report.json', 'eval-report.json', 'inference.md', 'manifest.json', 'model-card.md', 'weights']);
  for (const file of manifest.files) {
    const bytes = await readFile(path.join(out, file.path));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), file.sha256, `digest mismatch for ${file.path}`);
  }
  assert.ok(manifest.files.some((f) => f.path === 'weights/base.json' && f.role === 'base'));
  assert.ok(manifest.files.some((f) => f.path === 'weights/adapter.json' && f.role === 'adapter'));
  assert.ok(manifest.files.some((f) => f.path === 'model-card.md' && f.role === 'doc'));
  assert.match(manifest.bundleDigest, /^[0-9a-f]{64}$/);

  // checksums.txt covers the manifest too, in sha256sum -c format
  const checksums = await readFile(path.join(out, 'checksums.txt'), 'utf8');
  for (const file of [...manifest.files.map((f) => f.path), 'manifest.json']) {
    assert.ok(checksums.includes(`  ${file}\n`), `checksums.txt missing ${file}`);
  }

  const card = await readFile(path.join(out, 'model-card.md'), 'utf8');
  assert.match(card, /# Model card: demo 1\.2\.3/);
  assert.match(card, /0\.900000/);
  assert.match(card, /62\.50%/);
  assert.match(card, /holdout_loss_improvement/);
  assert.match(card, /synthetic data generated by this/i);

  const inference = await readFile(path.join(out, 'inference.md'), 'utf8');
  assert.match(inference, /sha256sum -c checksums\.txt/);
  assert.match(inference, /attachAdapter: 'weights\/adapter\.json'/);
  assert.match(inference, /roll back/i);
  assert.match(inference, /`weights\/base\.json` alone/);
});

test('exportModel refuses to package a failed evaluation', async () => {
  const { dir, artifacts } = await fixture();
  const out = path.join(dir, 'blocked');
  const failed = passedReport({
    verdict: { status: 'failed', improvementClaimAllowed: false, deliveryEligible: false, summary: 'gates failed' },
    gates: { passed: false, failed: [{ id: 'exact_match', reason: 'below threshold' }], results: [], summary: '4/5' },
  });
  await assert.rejects(() => exportModel({ outDir: out, artifacts, evalReport: failed }), (err) => {
    assert.equal(err.code, 'EXPORT_BLOCKED');
    assert.match(err.message, /exact_match/);
    return true;
  });
  await assert.rejects(() => readdir(out), /ENOENT/); // nothing was written
});

test('exportModel reports a full tuned checkpoint and its rollback path', async () => {
  const { dir, artifacts } = await fixture();
  const full = { ...artifacts, adapterCheckpoint: null, tunedCheckpoint: path.join(dir, 'run', 'adapter.json') };
  const result = await exportModel({ outDir: path.join(dir, 'full'), artifacts: full, evalReport: passedReport(), config: { project: 'demo', mode: 'full' } });
  assert.equal(result.manifest.kind, 'full');
  assert.ok(result.manifest.files.some((f) => f.path === 'weights/tuned.json'));
  assert.match(result.manifest.rollback, /weights\/base\.json/);
  const inference = await readFile(path.join(result.outDir, 'inference.md'), 'utf8');
  assert.match(inference, /full tuned checkpoint/);
});

test('exportModel inlines an eval report that has no file on disk', async () => {
  const { dir, artifacts } = await fixture();
  const result = await exportModel({ outDir: path.join(dir, 'inline'), artifacts, evalReport: passedReport() });
  const inlined = result.manifest.files.find((f) => f.path === 'eval-report.json');
  assert.equal(inlined.source, 'inline');
  const parsed = JSON.parse(await readFile(path.join(result.outDir, 'eval-report.json'), 'utf8'));
  assert.equal(parsed.verdict.status, 'passed');
});
