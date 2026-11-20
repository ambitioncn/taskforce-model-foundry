// Independent evaluation of a trained checkpoint.
//
// "Independent" means two concrete things. First, the evaluator only ever
// reads the holdout and validation splits, and it refuses to run at all if a
// holdout record also appears in train: a leaked holdout makes every number
// below meaningless. Second, it scores a model that was loaded back from a
// checkpoint file rather than the in-memory handle the trainer just finished
// with, so what gets measured is what would actually ship.

import { relativeImprovement, fieldAccuracy, characterF1, normalizedExactMatchScore } from './metrics.mjs';
import { sha256 } from '../core/hash.mjs';

export const EVAL_REPORT_FORMAT = 'tmf-eval-report-v1';

export function recordFingerprint(record) {
  // Corpus records come in two shapes: supervised pairs (prompt/completion)
  // and plain text. Keying only on the pair made every text-only record
  // fingerprint as "undefined|undefined", which reported the whole holdout as
  // leaked, so fall back to the text and refuse a record with neither.
  const prompt = record?.prompt ?? null;
  const completion = record?.completion ?? null;
  if (prompt !== null || completion !== null) return sha256(`pair:${prompt}|${completion}`);
  const text = record?.text ?? null;
  if (text !== null) return sha256(`text:${text}`);
  throw new Error('recordFingerprint needs a record with prompt/completion or text');
}

// Leakage is a gate, not a warning: a holdout set that overlaps train cannot
// be used to claim improvement, so this throws instead of annotating.
export function assertHoldoutIndependence({ holdout = [], train = [], validation = [] } = {}) {
  if (!holdout.length) {
    const err = new Error('evaluation needs a non-empty holdout split');
    err.code = 'HOLDOUT_EMPTY';
    throw err;
  }
  const trainPrints = new Set(train.map(recordFingerprint));
  const leaked = holdout.filter((r) => trainPrints.has(recordFingerprint(r)));
  if (leaked.length) {
    const err = new Error(
      `holdout is not independent: ${leaked.length}/${holdout.length} holdout records also appear in train`,
    );
    err.code = 'HOLDOUT_LEAKED';
    err.leaked = leaked.map((r) => recordFingerprint(r).slice(0, 12));
    throw err;
  }
  return {
    ok: true,
    holdout: holdout.length,
    train: train.length,
    validation: validation.length,
    holdoutFingerprint: sha256(holdout.map(recordFingerprint).sort().join('')),
  };
}

// Document-style records have no prompt/completion pair at all -- the data
// pipeline leaves both null for text-only rows -- so greedy decode can never
// score them. Keeping them in the denominator silently deflated accuracy (and
// fed the literal string 'null' to the tokenizer), so they are excluded here
// and reported separately instead.
export function isScorableRecord(record) {
  return typeof record?.prompt === 'string' && typeof record?.completion === 'string';
}

export async function exactMatchOnRecords(backend, handle, records, { maxNewTokens = 48, sampleLimit = 3 } = {}) {
  const scorable = records.filter(isScorableRecord);
  const excluded = records.length - scorable.length;
  let hit = 0;
  const samples = [];
  const predictions = [];
  for (const record of scorable) {
    const got = await backend.generate({ handle, prompt: record.prompt, maxNewTokens });
    if (got === record.completion) hit += 1;
    predictions.push({ predicted: got, expected: record.completion });
    if (samples.length < sampleLimit) samples.push({ prompt: record.prompt, got, want: record.completion });
  }
  // The gated number stays byte-identical exact match. The three below are
  // diagnostics only: with exact match pinned at 0 the report said nothing
  // about *how* wrong the completions were, which is the difference between a
  // formatting bug and an undertrained model.
  return {
    accuracy: scorable.length ? Number((hit / scorable.length).toFixed(6)) : null,
    hit,
    total: scorable.length,
    considered: records.length,
    excluded,
    normalized: predictions.length ? normalizedExactMatchScore(predictions) : null,
    fields: predictions.length ? fieldAccuracy(predictions) : null,
    characterF1: predictions.length ? characterF1(predictions) : null,
    samples,
  };
}

// One model, both splits, plus greedy-decode exact match on holdout.
export async function evaluateSplitSet(backend, handle, { encoded, records = {}, maxNewTokens = 48 } = {}) {
  if (!encoded?.holdout?.length) throw new Error('evaluateSplitSet needs encoded holdout examples');
  const holdout = await backend.evaluate({ handle, examples: encoded.holdout });
  const exactMatch = records.holdout?.some(isScorableRecord)
    ? await exactMatchOnRecords(backend, handle, records.holdout, { maxNewTokens })
    : null;
  const validation = encoded.validation?.length
    ? await backend.evaluate({ handle, examples: encoded.validation })
    : null;
  return { holdout: { ...holdout, exactMatch }, validation };
}

export function computeImprovement(baseline, tuned) {
  const baseExact = baseline.holdout.exactMatch?.accuracy ?? null;
  const tunedExact = tuned.holdout.exactMatch?.accuracy ?? null;
  return {
    holdoutLossRelative: relativeImprovement(baseline.holdout.loss, tuned.holdout.loss),
    holdoutLossAbsolute: Number((baseline.holdout.loss - tuned.holdout.loss).toFixed(6)),
    validationLossRelative: baseline.validation && tuned.validation
      ? relativeImprovement(baseline.validation.loss, tuned.validation.loss)
      : null,
    exactMatchAbsolute: baseExact === null || tunedExact === null
      ? null
      : Number((tunedExact - baseExact).toFixed(6)),
  };
}

export function assembleMetrics({ baseline, tuned }) {
  if (!tuned) return { baseline, tuned: null, improvement: null };
  return { baseline, tuned, improvement: computeImprovement(baseline, tuned) };
}
