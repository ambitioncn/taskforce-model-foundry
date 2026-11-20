// Threshold gates.
//
// The gate is the thing that stops an unimproved model from being called an
// improvement. It is evaluated against the measured metrics object and returns
// a per-rule verdict; `assertGatesPass` is what export and delivery paths call,
// and a failure there is terminal rather than advisory.

export const COMPARATORS = {
  gte: { test: (a, b) => a >= b, label: '>=' },
  lte: { test: (a, b) => a <= b, label: '<=' },
  gt: { test: (a, b) => a > b, label: '>' },
  lt: { test: (a, b) => a < b, label: '<' },
};

// Exact match is a proportion, and a proportion over a handful of records is
// not a measurement. train-ws-full scored it on the 2 supervised pairs that
// landed in a 25-record holdout, where a single lucky completion reads as 0.5
// and clears the 0.25 threshold outright. A gate whose sample count is below
// this floor is reported as unmeasured and fails; the floor constrains the
// denominator only -- no threshold moves -- and 10 samples give the estimate
// the same 0.1 resolution as the improvement threshold it is compared with.
export const MIN_EXACT_MATCH_SAMPLES = 10;

export const DEFAULT_GATES = [
  { id: 'holdout_loss_improvement', metric: 'improvement.holdoutLossRelative', comparator: 'gte', threshold: 0.3, description: 'post-training must cut holdout loss by at least 30% relative to the baseline' },
  { id: 'holdout_loss_absolute', metric: 'tuned.holdout.loss', comparator: 'lte', threshold: 1.5, description: 'absolute holdout loss ceiling' },
  { id: 'exact_match', metric: 'tuned.holdout.exactMatch.accuracy', comparator: 'gte', threshold: 0.25, sampleMetric: 'tuned.holdout.exactMatch.total', minSamples: MIN_EXACT_MATCH_SAMPLES, description: 'greedy-decoded completions must match exactly on at least a quarter of the holdout set' },
  { id: 'exact_match_improvement', metric: 'improvement.exactMatchAbsolute', comparator: 'gte', threshold: 0.1, sampleMetric: 'tuned.holdout.exactMatch.total', minSamples: MIN_EXACT_MATCH_SAMPLES, description: 'exact match must improve over the baseline by at least 10 points' },
  { id: 'no_validation_regression', metric: 'improvement.validationLossRelative', comparator: 'gte', threshold: -0.05, description: 'validation loss must not regress by more than 5% relative' },
];

export function readMetric(metrics, dottedPath) {
  return dottedPath.split('.').reduce((acc, key) => (acc === null || acc === undefined ? undefined : acc[key]), metrics);
}

export function evaluateGates(metrics, gates = DEFAULT_GATES) {
  const results = gates.map((gate) => {
    const comparator = COMPARATORS[gate.comparator];
    if (!comparator) throw new Error(`unknown gate comparator "${gate.comparator}" on gate "${gate.id}"`);
    const actual = readMetric(metrics, gate.metric);
    if (actual === undefined || actual === null || Number.isNaN(actual)) {
      return { ...gate, actual: actual ?? null, ok: false, reason: `metric "${gate.metric}" is missing from the evaluation report` };
    }
    if (gate.minSamples) {
      const samples = gate.sampleMetric ? readMetric(metrics, gate.sampleMetric) : undefined;
      if (typeof samples !== 'number' || Number.isNaN(samples)) {
        return {
          ...gate,
          actual: typeof actual === 'number' ? Number(actual.toFixed(6)) : actual,
          samples: null,
          ok: false,
          reason: `sample count "${gate.sampleMetric}" is missing, so ${gate.metric}=${actual} cannot be trusted`,
        };
      }
      if (samples < gate.minSamples) {
        return {
          ...gate,
          actual: typeof actual === 'number' ? Number(actual.toFixed(6)) : actual,
          samples,
          ok: false,
          reason: `${gate.metric}=${actual} was measured on ${samples} scorable holdout record(s), below the minimum of ${gate.minSamples}; the estimate is not usable`,
        };
      }
    }
    const ok = comparator.test(actual, gate.threshold);
    return {
      ...gate,
      actual: typeof actual === 'number' ? Number(actual.toFixed(6)) : actual,
      ...(gate.minSamples ? { samples: readMetric(metrics, gate.sampleMetric) ?? null } : {}),
      ok,
      reason: ok ? null : `${gate.metric}=${actual} fails ${comparator.label} ${gate.threshold}`,
    };
  });
  const failed = results.filter((r) => !r.ok);
  return {
    passed: failed.length === 0,
    results,
    failed: failed.map((r) => ({ id: r.id, reason: r.reason })),
    summary: `${results.length - failed.length}/${results.length} gates passed`,
  };
}

export function assertGatesPass(metrics, gates = DEFAULT_GATES) {
  const verdict = evaluateGates(metrics, gates);
  if (!verdict.passed) {
    const err = new Error(
      `evaluation gates failed (${verdict.summary}): ${verdict.failed.map((f) => f.reason).join('; ')}`,
    );
    err.code = 'GATES_FAILED';
    err.verdict = verdict;
    throw err;
  }
  return verdict;
}

// Regression check against a previously recorded evaluation of the same task.
export function checkRegression(current, previous, { tolerance = 0.02 } = {}) {
  if (!previous) return { ok: true, checked: false, reason: 'no previous evaluation recorded' };
  const findings = [];
  const currentLoss = readMetric(current, 'tuned.holdout.loss');
  const previousLoss = readMetric(previous, 'tuned.holdout.loss');
  if (typeof currentLoss === 'number' && typeof previousLoss === 'number') {
    const allowed = previousLoss * (1 + tolerance);
    if (currentLoss > allowed) {
      findings.push(`holdout loss regressed from ${previousLoss} to ${currentLoss} (allowed up to ${allowed.toFixed(6)})`);
    }
  }
  const currentEm = readMetric(current, 'tuned.holdout.exactMatch.accuracy');
  const previousEm = readMetric(previous, 'tuned.holdout.exactMatch.accuracy');
  if (typeof currentEm === 'number' && typeof previousEm === 'number' && currentEm < previousEm - tolerance) {
    findings.push(`exact match regressed from ${previousEm} to ${currentEm}`);
  }
  return { ok: findings.length === 0, checked: true, findings, tolerance };
}
