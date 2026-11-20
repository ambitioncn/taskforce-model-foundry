// Deterministic splitting plus leakage detection.
//
// Assignment is a hash of the record's canonical key, not its position, so the
// same record always lands in the same split no matter what order it was
// ingested in. That is what makes a re-run reproducible and makes "the holdout
// set never saw training" checkable rather than assumed.

import { sha256 } from '../core/hash.mjs';
import { canonicalKey } from './clean.mjs';
import { NearDuplicateIndex, DEFAULT_DEDUPE_CONFIG } from './dedupe.mjs';

export const SPLIT_NAMES = ['train', 'validation', 'holdout'];

// Kept equal to eval's MIN_EXACT_MATCH_SAMPLES on purpose: the data stage must
// not hand evaluation a holdout that evaluation will refuse to score.
export const MIN_SCORABLE_HOLDOUT = 10;

export const DEFAULT_SPLIT_CONFIG = {
  train: 0.7,
  validation: 0.15,
  holdout: 0.15,
  seed: 'tmf-split-v1',
  requireNonEmptyHoldout: true,
  // A corpus mixes supervised pairs with plain documents, and only the pairs
  // can be greedy-decoded and scored for exact match. Bucketing the mixed
  // population by hash alone let the pairs land almost entirely in train:
  // train-ws-full put 2 of ~40 pairs in a 25-record holdout, so exact match
  // was an estimate over two samples. Strata are split independently and
  // proportionally instead, so every holdout keeps its share of pairs.
  stratify: true,
  requireScorableHoldout: true,
  // Stratification keeps the share, but a share of too few pairs is still too
  // few: an exact-match rate over 2 samples is noise. This is the data-side
  // half of eval's MIN_EXACT_MATCH_SAMPLES -- a corpus that cannot fill a
  // scorable holdout fails here, while the job is cheap, instead of producing
  // an unusable estimate after training. Unit fixtures that only exercise
  // assignment pass 0 explicitly.
  minScorableHoldout: MIN_SCORABLE_HOLDOUT,
};

function bucket(text, seed) {
  // 32-bit prefix of the digest, mapped to [0,1).
  const digest = sha256(`${seed}:${canonicalKey(text)}`);
  return parseInt(digest.slice(0, 8), 16) / 0x100000000;
}

// A record is scorable when it carries a supervised pair; text-only rows can
// contribute loss but never exact match. This mirrors eval's isScorableRecord.
export function splitStratum(record) {
  return typeof record?.prompt === 'string' && typeof record?.completion === 'string' ? 'pair' : 'text';
}

function assignByRatio(fraction, cfg) {
  if (fraction < cfg.train) return 'train';
  if (fraction < cfg.train + cfg.validation) return 'validation';
  return 'holdout';
}

// Proportional allocation inside one stratum. The hash still decides which
// record goes where -- records are ranked by their bucket value, not by input
// position -- but the rank, rather than the raw value, is compared against the
// ratios. That makes the per-stratum counts exact instead of binomial, which
// is what keeps a small pair population represented in holdout.
function assignStratum(records, cfg) {
  const ranked = records
    .map((record) => ({ record, b: bucket(record.text, cfg.seed), key: canonicalKey(record.text) }))
    .sort((a, b) => (a.b - b.b) || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  return ranked.map((entry, index) => ({
    record: entry.record,
    name: assignByRatio((index + 0.5) / ranked.length, cfg),
  }));
}

export function splitRecords(records, config = {}) {
  const cfg = { ...DEFAULT_SPLIT_CONFIG, ...config };
  const total = cfg.train + cfg.validation + cfg.holdout;
  if (Math.abs(total - 1) > 1e-6) {
    throw new Error(`split ratios must sum to 1, got ${total} (train=${cfg.train}, validation=${cfg.validation}, holdout=${cfg.holdout})`);
  }
  for (const name of SPLIT_NAMES) {
    if (cfg[name] < 0) throw new Error(`split ratio "${name}" must not be negative`);
  }
  const splits = { train: [], validation: [], holdout: [] };
  if (cfg.stratify) {
    const strata = new Map();
    for (const record of records) {
      const stratum = splitStratum(record);
      if (!strata.has(stratum)) strata.set(stratum, []);
      strata.get(stratum).push(record);
    }
    for (const stratum of [...strata.keys()].sort()) {
      for (const { record, name } of assignStratum(strata.get(stratum), cfg)) {
        splits[name].push({ ...record, split: name, stratum });
      }
    }
  } else {
    for (const record of records) {
      const name = assignByRatio(bucket(record.text, cfg.seed), cfg);
      splits[name].push({ ...record, split: name, stratum: splitStratum(record) });
    }
  }
  if (cfg.requireNonEmptyHoldout && splits.holdout.length === 0) {
    throw new Error('holdout split is empty; evaluation would have nothing independent to measure against');
  }
  // A holdout with no supervised pair cannot produce an exact-match number at
  // all, which is a silent hole in evaluation rather than a bad score.
  if (cfg.requireScorableHoldout && cfg.holdout > 0) {
    const pairsAvailable = records.some((r) => splitStratum(r) === 'pair');
    const pairsHeldOut = splits.holdout.some((r) => splitStratum(r) === 'pair');
    if (pairsAvailable && !pairsHeldOut) {
      throw new Error('holdout split contains no supervised pair; exact match would have nothing to score');
    }
    const minScorable = cfg.minScorableHoldout ?? 0;
    const scorable = splits.holdout.filter((r) => splitStratum(r) === 'pair').length;
    if (pairsAvailable && scorable < minScorable) {
      const pairs = records.filter((r) => splitStratum(r) === 'pair').length;
      const needed = Math.ceil(minScorable / cfg.holdout);
      throw new Error(
        `holdout has ${scorable} scorable pair(s), below the minimum ${minScorable}: `
        + `the corpus carries ${pairs} supervised pair(s) and a ${cfg.holdout} holdout share, `
        + `so exact match would be an estimate over too few samples. `
        + `Supply at least ${needed} supervised pair(s), or set split.minScorableHoldout explicitly to accept the smaller sample.`,
      );
    }
  }
  return splits;
}

// Cross-split contamination check. Exact matches are canonical-key collisions;
// near matches are MinHash/Jaccard hits above the dedupe threshold.
export function checkLeakage(splits, { dedupe = {}, checkNear = true } = {}) {
  const cfg = { ...DEFAULT_DEDUPE_CONFIG, ...dedupe };
  const exact = [];
  const near = [];

  const keyIndex = new Map();
  for (const name of SPLIT_NAMES) {
    for (const record of splits[name] ?? []) {
      const key = canonicalKey(record.text);
      if (!keyIndex.has(key)) keyIndex.set(key, []);
      keyIndex.get(key).push({ split: name, id: record.id });
    }
  }
  for (const [key, hits] of keyIndex) {
    const splitsSeen = [...new Set(hits.map((h) => h.split))];
    if (splitsSeen.length > 1) {
      exact.push({ kind: 'exact', splits: splitsSeen.sort(), ids: hits.map((h) => h.id), keyPreview: key.slice(0, 48) });
    }
  }

  if (checkNear) {
    // Index train+validation, then probe the holdout against it: the direction
    // that matters is "did anything the model trained on reappear in holdout".
    const index = new NearDuplicateIndex(cfg);
    for (const name of ['train', 'validation']) {
      for (const record of splits[name] ?? []) index.add(`${name}:${record.id}`, record.text);
    }
    for (const record of splits.holdout ?? []) {
      const { match } = index.query(record.text);
      if (match) {
        near.push({
          kind: 'near',
          holdoutId: record.id,
          matchedId: match.id,
          similarity: Number(match.similarity.toFixed(4)),
          threshold: cfg.threshold,
        });
      }
    }
  }

  const findings = [...exact, ...near];
  return {
    ok: findings.length === 0,
    exact,
    near,
    counts: { exact: exact.length, near: near.length },
    checked: { train: splits.train?.length ?? 0, validation: splits.validation?.length ?? 0, holdout: splits.holdout?.length ?? 0 },
  };
}

export function assertNoLeakage(splits, options = {}) {
  const result = checkLeakage(splits, options);
  if (!result.ok) {
    const parts = [];
    if (result.exact.length) parts.push(`${result.exact.length} exact overlap(s) across splits`);
    if (result.near.length) parts.push(`${result.near.length} near-duplicate holdout record(s)`);
    const err = new Error(`split leakage check failed: ${parts.join('; ')}`);
    err.leakage = result;
    throw err;
  }
  return result;
}
