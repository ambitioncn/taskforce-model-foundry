import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseKeyValues, exactMatchScore, fieldAccuracy, characterF1, relativeImprovement,
  normalizeCompletion, normalizedExactMatchScore,
} from '../src/eval/metrics.mjs';
import {
  COMPARATORS, DEFAULT_GATES, MIN_EXACT_MATCH_SAMPLES, readMetric, evaluateGates, assertGatesPass, checkRegression,
} from '../src/eval/gates.mjs';
import { exactMatchOnRecords } from '../src/eval/evaluator.mjs';

// ---------------------------------------------------------------- metrics

test('parseKeyValues reads key=value lines and ignores prose', () => {
  const parsed = parseKeyValues('queue=infra\npriority = P1 \nthis is not a field\n_owner=on-call\nbad key=1');
  assert.deepEqual(parsed, { queue: 'infra', priority: 'P1', _owner: 'on-call' });
});

test('parseKeyValues tolerates null and empty input', () => {
  assert.deepEqual(parseKeyValues(null), {});
  assert.deepEqual(parseKeyValues(''), {});
});

test('exactMatchScore counts only byte-identical completions', () => {
  const score = exactMatchScore([
    { predicted: 'queue=infra', expected: 'queue=infra' },
    { predicted: 'queue=infra ', expected: 'queue=infra' },
    { predicted: 'queue=identity', expected: 'queue=infra' },
    { predicted: 'queue=net', expected: 'queue=net' },
  ]);
  assert.deepEqual(score, { hit: 2, total: 4, accuracy: 0.5 });
});

test('exactMatchScore reports 0 rather than NaN on an empty prediction set', () => {
  assert.deepEqual(exactMatchScore([]), { hit: 0, total: 0, accuracy: 0 });
});

test('fieldAccuracy separates a right queue from a wrong priority', () => {
  const fields = fieldAccuracy([
    { predicted: 'queue=infra\npriority=P2', expected: 'queue=infra\npriority=P1' },
    { predicted: 'queue=infra\npriority=P1', expected: 'queue=infra\npriority=P1' },
  ]);
  assert.deepEqual(fields, [
    { field: 'priority', hit: 1, total: 2, accuracy: 0.5 },
    { field: 'queue', hit: 2, total: 2, accuracy: 1 },
  ]);
});

test('fieldAccuracy scores a missing predicted field as a miss, never as absent', () => {
  const fields = fieldAccuracy([{ predicted: 'queue=infra', expected: 'queue=infra\npriority=P1' }]);
  const priority = fields.find((f) => f.field === 'priority');
  assert.deepEqual(priority, { field: 'priority', hit: 0, total: 1, accuracy: 0 });
});

test('fieldAccuracy ignores fields the model invented', () => {
  const fields = fieldAccuracy([{ predicted: 'queue=infra\nextra=noise', expected: 'queue=infra' }]);
  assert.deepEqual(fields.map((f) => f.field), ['queue']);
});

test('characterF1 is 1 on an exact match and 0 on a disjoint prediction', () => {
  assert.equal(characterF1([{ predicted: 'abc', expected: 'abc' }]).f1, 1);
  assert.equal(characterF1([{ predicted: 'xyz', expected: 'abc' }]).f1, 0);
});

test('characterF1 counts a repeated character only as often as it is wanted', () => {
  const m = characterF1([{ predicted: 'aaaa', expected: 'aa' }]);
  assert.equal(m.precision, 0.5); // 2 of 4 predicted characters are credited
  assert.equal(m.recall, 1);
  assert.ok(Math.abs(m.f1 - 2 / 3) < 1e-6);
});

test('characterF1 places a near miss strictly between noise and exact', () => {
  const near = characterF1([{ predicted: 'queue=infr', expected: 'queue=infra' }]).f1;
  assert.ok(near > 0.9 && near < 1, `near miss should be high but not perfect: ${near}`);
});

test('relativeImprovement treats loss as lower-is-better and accuracy as higher-is-better', () => {
  assert.equal(relativeImprovement(4, 1), 0.75);
  assert.equal(relativeImprovement(1, 4), -3);
  assert.equal(relativeImprovement(0.2, 0.4, { lowerIsBetter: false }), 1);
});

test('relativeImprovement returns null on missing numbers and is total at a zero baseline', () => {
  assert.equal(relativeImprovement(null, 1), null);
  assert.equal(relativeImprovement(1, undefined), null);
  assert.equal(relativeImprovement(0, 0), 0);
  assert.equal(relativeImprovement(0, 0.5), -Infinity);
  assert.equal(relativeImprovement(0, 0.5, { lowerIsBetter: false }), Infinity);
});

// ---------------------------------------------------------------- gates

const passingMetrics = {
  // `total` is the scorable-record count the evaluator already reports; the
  // exact-match gates refuse to read an accuracy without it.
  tuned: { holdout: { loss: 0.222503, exactMatch: { accuracy: 0.3, total: 24 } } },
  improvement: { holdoutLossRelative: 0.95, exactMatchAbsolute: 0.3, validationLossRelative: 0.9 },
};

test('COMPARATORS implement the four threshold directions', () => {
  assert.equal(COMPARATORS.gte.test(1, 1), true);
  assert.equal(COMPARATORS.gt.test(1, 1), false);
  assert.equal(COMPARATORS.lte.test(1, 1), true);
  assert.equal(COMPARATORS.lt.test(1, 1), false);
});

test('readMetric walks a dotted path and returns undefined instead of throwing', () => {
  assert.equal(readMetric(passingMetrics, 'tuned.holdout.loss'), 0.222503);
  assert.equal(readMetric(passingMetrics, 'tuned.missing.loss'), undefined);
  assert.equal(readMetric({}, 'a.b.c'), undefined);
});

test('evaluateGates passes a genuinely improved model', () => {
  const verdict = evaluateGates(passingMetrics);
  assert.equal(verdict.passed, true);
  assert.equal(verdict.failed.length, 0);
  assert.equal(verdict.summary, `${DEFAULT_GATES.length}/${DEFAULT_GATES.length} gates passed`);
});

test('evaluateGates fails an unimproved model and names every broken rule', () => {
  const verdict = evaluateGates({
    tuned: { holdout: { loss: 4.5, exactMatch: { accuracy: 0, total: 20 } } },
    improvement: { holdoutLossRelative: 0.01, exactMatchAbsolute: 0, validationLossRelative: -0.4 },
  });
  assert.equal(verdict.passed, false);
  assert.deepEqual(verdict.failed.map((f) => f.id).sort(), [
    'exact_match', 'exact_match_improvement', 'holdout_loss_absolute',
    'holdout_loss_improvement', 'no_validation_regression',
  ]);
  for (const f of verdict.failed) assert.match(f.reason, /fails [<>]=? /);
});

test('a missing metric fails its gate instead of silently passing', () => {
  const verdict = evaluateGates({}, [DEFAULT_GATES[0]]);
  assert.equal(verdict.passed, false);
  assert.match(verdict.failed[0].reason, /is missing from the evaluation report/);
});

test('a NaN metric fails its gate', () => {
  const verdict = evaluateGates({ tuned: { holdout: { loss: NaN } } }, [
    { id: 'l', metric: 'tuned.holdout.loss', comparator: 'lte', threshold: 1.5 },
  ]);
  assert.equal(verdict.passed, false);
});

test('an unknown comparator is a configuration error, not a soft failure', () => {
  assert.throws(
    () => evaluateGates(passingMetrics, [{ id: 'bogus', metric: 'tuned.holdout.loss', comparator: 'approx', threshold: 1 }]),
    /unknown gate comparator "approx" on gate "bogus"/,
  );
});

test('assertGatesPass returns the verdict on success and throws GATES_FAILED on failure', () => {
  assert.equal(assertGatesPass(passingMetrics).passed, true);
  assert.throws(() => assertGatesPass({}), (err) => {
    assert.equal(err.code, 'GATES_FAILED');
    assert.equal(err.verdict.passed, false);
    assert.match(err.message, /evaluation gates failed/);
    return true;
  });
});

test('checkRegression is explicit about having no baseline to compare against', () => {
  const r = checkRegression(passingMetrics, null);
  assert.equal(r.ok, true);
  assert.equal(r.checked, false);
});

test('checkRegression allows noise inside tolerance but catches a real regression', () => {
  const previous = { tuned: { holdout: { loss: 1, exactMatch: { accuracy: 0.5 } } } };
  const withinNoise = { tuned: { holdout: { loss: 1.01, exactMatch: { accuracy: 0.49 } } } };
  const regressed = { tuned: { holdout: { loss: 1.5, exactMatch: { accuracy: 0.2 } } } };
  assert.equal(checkRegression(withinNoise, previous).ok, true);
  const bad = checkRegression(regressed, previous);
  assert.equal(bad.ok, false);
  assert.equal(bad.findings.length, 2);
  assert.match(bad.findings.join(' '), /holdout loss regressed/);
  assert.match(bad.findings.join(' '), /exact match regressed/);
});

test('exactMatchOnRecords excludes unscorable document records from the denominator', async () => {
  const backend = {
    async generate({ prompt }) {
      return prompt === 'q1' ? 'a1' : 'wrong';
    },
  };
  const records = [
    { prompt: 'q1', completion: 'a1' },
    { prompt: 'q2', completion: 'a2' },
    { prompt: null, completion: null, text: 'doc body' },
    { prompt: null, completion: null, text: 'another doc' },
  ];
  const got = await exactMatchOnRecords(backend, 'h', records);
  assert.equal(got.hit, 1);
  assert.equal(got.total, 2);
  assert.equal(got.considered, 4);
  assert.equal(got.excluded, 2);
  assert.equal(got.accuracy, 0.5);
});

test('exactMatchOnRecords reports null accuracy when nothing is scorable', async () => {
  const backend = { async generate() { throw new Error('must not generate on unscorable records'); } };
  const got = await exactMatchOnRecords({ ...backend }, 'h', [{ prompt: null, completion: null }]);
  assert.equal(got.accuracy, null);
  assert.equal(got.total, 0);
  assert.equal(got.excluded, 1);
});


// The train-ws-full report scored exact match on 2 records. 1/2 would read as
// 0.5 and clear the 0.25 threshold, so the denominator has to gate too.
test('exact-match gates refuse an estimate measured on too few records', () => {
  const thin = {
    tuned: { holdout: { loss: 0.222503, exactMatch: { accuracy: 0.5, hit: 1, total: 2 } } },
    improvement: { holdoutLossRelative: 0.95, exactMatchAbsolute: 0.5, validationLossRelative: 0.9 },
  };
  const verdict = evaluateGates(thin);
  assert.equal(verdict.passed, false);
  const ids = verdict.failed.map((f) => f.id).sort();
  assert.deepEqual(ids, ['exact_match', 'exact_match_improvement']);
  for (const failure of verdict.failed) {
    assert.match(failure.reason, /measured on 2 scorable holdout record\(s\), below the minimum of 10/);
  }
  // The thresholds themselves are untouched.
  const em = DEFAULT_GATES.find((g) => g.id === 'exact_match');
  assert.equal(em.threshold, 0.25);
  assert.equal(em.minSamples, MIN_EXACT_MATCH_SAMPLES);
  assert.equal(verdict.results.find((r) => r.id === 'exact_match').samples, 2);
});

test('exact-match gates refuse a report with no scorable-sample count at all', () => {
  const noTotal = {
    tuned: { holdout: { loss: 0.222503, exactMatch: { accuracy: 0.9 } } },
    improvement: { holdoutLossRelative: 0.95, exactMatchAbsolute: 0.9, validationLossRelative: 0.9 },
  };
  const verdict = evaluateGates(noTotal);
  assert.equal(verdict.passed, false);
  assert.match(verdict.failed.find((f) => f.id === 'exact_match').reason, /sample count .* is missing/);
});

test('a sufficient scorable sample lets the exact-match gates be decided on their thresholds', () => {
  const enough = {
    tuned: { holdout: { loss: 0.222503, exactMatch: { accuracy: 0.3, hit: 3, total: 10 } } },
    improvement: { holdoutLossRelative: 0.95, exactMatchAbsolute: 0.3, validationLossRelative: 0.9 },
  };
  assert.equal(evaluateGates(enough).passed, true);
  const below = {
    tuned: { holdout: { loss: 0.222503, exactMatch: { accuracy: 0.2, hit: 2, total: 10 } } },
    improvement: { holdoutLossRelative: 0.95, exactMatchAbsolute: 0.2, validationLossRelative: 0.9 },
  };
  assert.match(evaluateGates(below).failed.find((f) => f.id === 'exact_match').reason, /fails >= 0.25/);
});

test('normalizedExactMatchScore forgives whitespace but not content', () => {
  const score = normalizedExactMatchScore([
    { predicted: '  queue=infra\n\npriority=P1  ', expected: 'queue=infra\npriority=P1' },
    { predicted: 'queue=infra\npriority=P2', expected: 'queue=infra\npriority=P1' },
  ]);
  assert.deepEqual(score, { hit: 1, total: 2, accuracy: 0.5 });
});

test('normalizeCompletion collapses runs of inner whitespace', () => {
  assert.equal(normalizeCompletion('queue   =   infra\n \n x'), 'queue = infra\nx');
});

test('exactMatchOnRecords reports field, character and normalized diagnostics alongside the gated number', async () => {
  const backend = { generate: async () => ' queue=infra\npriority=P2 ' };
  const got = await exactMatchOnRecords(backend, {}, [
    { prompt: 'a', completion: 'queue=infra\npriority=P1' },
  ]);
  assert.equal(got.accuracy, 0, 'gated exact match stays byte-identical');
  assert.equal(got.normalized.accuracy, 0, 'wrong priority is still a miss after normalization');
  assert.deepEqual(got.fields.map((f) => [f.field, f.accuracy]), [['priority', 0], ['queue', 1]]);
  assert.ok(got.characterF1.f1 > 0.8 && got.characterF1.f1 < 1, `near miss, got ${got.characterF1.f1}`);
});
