// Export must be able to reuse the exact evaluation report a human reviewed,
// and must refuse one whose contents no longer match its own digest.
import test from 'node:test';
import assert from 'node:assert/strict';

import { sha256Json } from '../src/core/hash.mjs';
import { assertReportIntegrity } from '../src/jobs/evaluate.mjs';
import { EVAL_REPORT_FORMAT } from '../src/eval/evaluator.mjs';

function sealedReport(overrides = {}) {
  const report = {
    format: EVAL_REPORT_FORMAT,
    startedAt: '2026-11-18T14:28:22.392Z',
    finishedAt: '2026-11-18T14:29:01.105Z',
    lineage: { jobId: 'train-demo', mode: 'full-finetune' },
    verdict: { status: 'passed', improvementClaimAllowed: true, deliveryEligible: true },
    ...overrides,
  };
  report.reportDigest = sha256Json({ ...report, reportDigest: undefined });
  return report;
}

test('a sealed report passes integrity and job-binding checks', () => {
  const report = sealedReport();
  assert.equal(assertReportIntegrity(report, { jobId: 'train-demo' }), report);
});

test('an in-memory reportFile annotation does not break the digest', () => {
  // runEvaluationJob attaches reportFile after writing, so the same object
  // round-tripped through the API must still validate.
  const report = { ...sealedReport(), reportFile: '/tmp/run/eval-report.json' };
  assert.doesNotThrow(() => assertReportIntegrity(report));
});

test('an edited report is rejected instead of being bundled as lineage', () => {
  const report = sealedReport();
  report.verdict.status = 'failed';
  assert.throws(() => assertReportIntegrity(report), /digest mismatch/);
});

test('a report missing its digest is rejected', () => {
  const report = sealedReport();
  delete report.reportDigest;
  assert.throws(() => assertReportIntegrity(report), /missing reportDigest/);
});

test('a foreign document is rejected on format', () => {
  assert.throws(() => assertReportIntegrity({ format: 'something-else', reportDigest: 'x' }), /format mismatch/);
});

test('a report from another job cannot be pinned to this export', () => {
  const report = sealedReport({ lineage: { jobId: 'train-other' } });
  assert.throws(() => assertReportIntegrity(report, { jobId: 'train-demo' }), /belongs to job train-other/);
});
