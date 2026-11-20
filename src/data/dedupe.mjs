// Exact and near-duplicate detection.
//
// Exact duplicates are caught on the canonical key. Near duplicates use MinHash
// over character shingles with LSH banding, so the comparison count stays
// linear instead of quadratic as a corpus grows.

import { sha256 } from '../core/hash.mjs';
import { canonicalKey } from './clean.mjs';

export const DEFAULT_DEDUPE_CONFIG = {
  nearDuplicates: true,
  shingleSize: 5,
  numHashes: 64,
  bands: 16,
  threshold: 0.85,
};

function hash32(str, seed) {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  // Final avalanche so the seeded hashes decorrelate.
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h ^= h >>> 15;
  return h >>> 0;
}

export function shingles(text, size) {
  const key = canonicalKey(text);
  const chars = [...key];
  if (chars.length <= size) return new Set([key]);
  const out = new Set();
  for (let i = 0; i + size <= chars.length; i += 1) out.add(chars.slice(i, i + size).join(''));
  return out;
}

export function minhashSignature(text, { shingleSize, numHashes } = DEFAULT_DEDUPE_CONFIG) {
  const grams = shingles(text, shingleSize);
  const sig = new Uint32Array(numHashes).fill(0xffffffff);
  for (const gram of grams) {
    for (let h = 0; h < numHashes; h += 1) {
      const v = hash32(gram, (0x9e3779b9 + h * 0x85ebca6b) >>> 0);
      if (v < sig[h]) sig[h] = v;
    }
  }
  return sig;
}

export function signatureSimilarity(a, b) {
  let same = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === b[i]) same += 1;
  return same / a.length;
}

export function jaccard(textA, textB, shingleSize = DEFAULT_DEDUPE_CONFIG.shingleSize) {
  const a = shingles(textA, shingleSize);
  const b = shingles(textB, shingleSize);
  let inter = 0;
  for (const g of a) if (b.has(g)) inter += 1;
  const union = a.size + b.size - inter;
  return union === 0 ? 1 : inter / union;
}

function bandKeys(sig, bands) {
  const rows = Math.floor(sig.length / bands);
  const keys = [];
  for (let b = 0; b < bands; b += 1) {
    const slice = Array.from(sig.subarray(b * rows, (b + 1) * rows)).join(',');
    keys.push(`${b}:${hash32(slice, 0x1b873593)}`);
  }
  return keys;
}

export class NearDuplicateIndex {
  constructor(config = {}) {
    this.config = { ...DEFAULT_DEDUPE_CONFIG, ...config };
    if (this.config.numHashes % this.config.bands !== 0) {
      throw new Error(`numHashes ${this.config.numHashes} must be divisible by bands ${this.config.bands}`);
    }
    this.buckets = new Map();
    this.entries = [];
  }

  // Closest already-indexed entry at or above threshold, else null.
  query(text) {
    const sig = minhashSignature(text, this.config);
    const candidates = new Set();
    for (const key of bandKeys(sig, this.config.bands)) {
      for (const idx of this.buckets.get(key) ?? []) candidates.add(idx);
    }
    let best = null;
    for (const idx of candidates) {
      const entry = this.entries[idx];
      const approx = signatureSimilarity(sig, entry.sig);
      // Confirm LSH candidates with exact Jaccard so banding cannot force a merge.
      const exact = jaccard(text, entry.text, this.config.shingleSize);
      if (exact >= this.config.threshold && (!best || exact > best.similarity)) {
        best = { id: entry.id, similarity: exact, approximate: approx };
      }
    }
    return { signature: sig, match: best };
  }

  add(id, text, signature) {
    const sig = signature ?? minhashSignature(text, this.config);
    const idx = this.entries.length;
    this.entries.push({ id, text, sig });
    for (const key of bandKeys(sig, this.config.bands)) {
      if (!this.buckets.has(key)) this.buckets.set(key, []);
      this.buckets.get(key).push(idx);
    }
    return idx;
  }

  get size() {
    return this.entries.length;
  }
}

export function exactKey(text) {
  return sha256(canonicalKey(text));
}
