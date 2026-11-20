// Evaluation metrics.
//
// Loss and perplexity come from the same forward pass the trainer uses; the
// generation metrics decode greedily so a reported number is reproducible from
// the checkpoint alone.

export function parseKeyValues(text) {
  const out = {};
  for (const line of String(text ?? '').split('\n')) {
    const match = /^\s*([A-Za-z_][\w.-]*)\s*=\s*(.*?)\s*$/.exec(line);
    if (match) out[match[1]] = match[2];
  }
  return out;
}

export function exactMatchScore(predictions) {
  const hit = predictions.filter((p) => p.predicted === p.expected).length;
  return { hit, total: predictions.length, accuracy: predictions.length ? hit / predictions.length : 0 };
}

// Diagnostic normalization: the gate metric stays byte-identical, but a
// completion that differs only in surrounding whitespace or blank lines is a
// different failure from one that got the content wrong, and a report that
// cannot tell them apart cannot direct the next experiment.
export function normalizeCompletion(text) {
  return String(text ?? '')
    .split('\n')
    .map((line) => line.trim().replace(/\s+/g, ' '))
    .filter((line) => line.length > 0)
    .join('\n');
}

export function normalizedExactMatchScore(predictions) {
  const hit = predictions.filter(
    (p) => normalizeCompletion(p.predicted) === normalizeCompletion(p.expected),
  ).length;
  return { hit, total: predictions.length, accuracy: predictions.length ? Number((hit / predictions.length).toFixed(6)) : 0 };
}

// Per-field accuracy for structured completions: a model can get the queue
// right and the priority wrong, and the report should say so.
export function fieldAccuracy(predictions) {
  const fields = new Map();
  for (const p of predictions) {
    const got = parseKeyValues(p.predicted);
    const want = parseKeyValues(p.expected);
    for (const key of Object.keys(want)) {
      if (!fields.has(key)) fields.set(key, { field: key, hit: 0, total: 0 });
      const entry = fields.get(key);
      entry.total += 1;
      if (got[key] === want[key]) entry.hit += 1;
    }
  }
  return [...fields.values()]
    .map((e) => ({ ...e, accuracy: e.total ? Number((e.hit / e.total).toFixed(6)) : 0 }))
    .sort((a, b) => (a.field < b.field ? -1 : 1));
}

// Character-level similarity, so a near-miss is distinguishable from noise.
export function characterF1(predictions) {
  let tp = 0;
  let predLen = 0;
  let goldLen = 0;
  for (const p of predictions) {
    const got = String(p.predicted ?? '');
    const want = String(p.expected ?? '');
    predLen += got.length;
    goldLen += want.length;
    const counts = new Map();
    for (const ch of want) counts.set(ch, (counts.get(ch) ?? 0) + 1);
    for (const ch of got) {
      const left = counts.get(ch) ?? 0;
      if (left > 0) {
        tp += 1;
        counts.set(ch, left - 1);
      }
    }
  }
  const precision = predLen ? tp / predLen : 0;
  const recall = goldLen ? tp / goldLen : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision: Number(precision.toFixed(6)), recall: Number(recall.toFixed(6)), f1: Number(f1.toFixed(6)) };
}

export function relativeImprovement(baseline, candidate, { lowerIsBetter = true } = {}) {
  if (baseline === null || candidate === null || baseline === undefined || candidate === undefined) return null;
  if (baseline === 0) return candidate === 0 ? 0 : (lowerIsBetter ? -Infinity : Infinity);
  const raw = lowerIsBetter ? (baseline - candidate) / Math.abs(baseline) : (candidate - baseline) / Math.abs(baseline);
  return Number(raw.toFixed(6));
}
