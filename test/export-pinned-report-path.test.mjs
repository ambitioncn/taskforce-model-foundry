// Regression: hosts (Hermes / OpenClaw tool calls) can only pass JSON scalars,
// so `evalReport` reaches the API as a file path. It has to be read, parsed and
// integrity-checked exactly like an in-process report object.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp, writeFile, mkdir } from 'node:fs/promises';

import { exportModel } from '../src/api.mjs';
import { sha256Json } from '../src/core/hash.mjs';
import { EVAL_REPORT_FORMAT } from '../src/eval/evaluator.mjs';

function signed(report) {
  return { ...report, reportDigest: sha256Json({ ...report, reportDigest: undefined }) };
}

function report({ status = 'failed', deliveryEligible = false, passed = false } = {}) {
  return signed({
    format: EVAL_REPORT_FORMAT,
    jobId: 'job-1',
    lineage: { jobId: 'job-1' },
    verdict: { status, deliveryEligible },
    gates: { passed, failed: passed ? [] : [{ id: 'exact_match' }], results: [] },
    metrics: {},
  });
}

async function fixture() {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'tmf-pinned-'));
  await mkdir(path.join(dir, 'run'), { recursive: true });
  return dir;
}

const config = (dir) => ({ project: 'demo', mode: 'adapter', baseDir: dir, runRootAbsolute: path.join(dir, 'run') });

test('a pinned report path is read, parsed and integrity-checked', async () => {
  const dir = await fixture();
  const file = path.join(dir, 'eval-report.json');
  await writeFile(file, JSON.stringify(report({ status: 'failed' })));
  // Getting past the report check into artifact resolution (instead of the old
  // "eval report must be an object") proves the path was read, parsed and
  // integrity-checked exactly like an in-process report object.
  await assert.rejects(
    () => exportModel({ config: config(dir), baseDir: dir, jobId: 'job-1', evalReport: file, outDir: path.join(dir, 'dist'), evaluateIfMissing: false }),
    (err) => {
      assert.doesNotMatch(err.message, /eval report must be an object/);
      assert.match(err.message, /no durable job state/);
      return true;
    },
  );
});

test('a relative pinned report path resolves against the config baseDir', async () => {
  const dir = await fixture();
  await writeFile(path.join(dir, 'rel-report.json'), JSON.stringify(report({ status: 'failed' })));
  await assert.rejects(
    () => exportModel({ config: config(dir), baseDir: dir, jobId: 'job-1', evalReport: 'rel-report.json', outDir: path.join(dir, 'dist'), evaluateIfMissing: false }),
    (err) => {
      // resolved relative to baseDir, so it never reports a missing file
      assert.doesNotMatch(err.message, /eval report file not found/);
      assert.match(err.message, /no durable job state/);
      return true;
    },
  );
});

test('a tampered pinned report file is refused by the digest check', async () => {
  const dir = await fixture();
  const file = path.join(dir, 'tampered.json');
  const tampered = { ...report({ status: 'passed', deliveryEligible: true, passed: true }), verdict: { status: 'passed', deliveryEligible: true } };
  tampered.metrics = { exactMatch: 1 }; // edited after signing
  await writeFile(file, JSON.stringify(tampered));
  await assert.rejects(
    () => exportModel({ config: config(dir), baseDir: dir, evalReport: file, outDir: path.join(dir, 'dist'), evaluateIfMissing: false }),
    /digest mismatch/,
  );
});

test('a missing or malformed pinned report path fails with a path-specific error', async () => {
  const dir = await fixture();
  await assert.rejects(
    () => exportModel({ config: config(dir), baseDir: dir, evalReport: path.join(dir, 'nope.json'), outDir: path.join(dir, 'dist'), evaluateIfMissing: false }),
    /eval report file not found/,
  );
  const bad = path.join(dir, 'bad.json');
  await writeFile(bad, 'not json at all');
  await assert.rejects(
    () => exportModel({ config: config(dir), baseDir: dir, evalReport: bad, outDir: path.join(dir, 'dist'), evaluateIfMissing: false }),
    /not valid JSON/,
  );
  const arr = path.join(dir, 'arr.json');
  await writeFile(arr, '[1,2,3]');
  await assert.rejects(
    () => exportModel({ config: config(dir), baseDir: dir, evalReport: arr, outDir: path.join(dir, 'dist'), evaluateIfMissing: false }),
    /does not contain a report object/,
  );
});
