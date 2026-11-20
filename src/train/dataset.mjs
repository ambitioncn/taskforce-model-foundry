// Turns pipeline records into token sequences.
//
// Instruction records keep a prompt boundary so the loss is only taken on the
// completion; plain documents train on every position.

import { Tokenizer } from '../model/tokenizer.mjs';

export function buildTokenizer(records, { maxVocab = 256 } = {}) {
  return Tokenizer.fromTexts(records.map((r) => (r.prompt !== null && r.prompt !== undefined ? `${r.prompt}${r.completion ?? ''}` : r.text)), { maxVocab });
}

export function encodeRecord(record, tokenizer, { maxSeqLen = 96 } = {}) {
  const hasPrompt = record.prompt !== null && record.prompt !== undefined && record.completion !== null && record.completion !== undefined;
  let ids;
  let promptLen = 0;
  if (hasPrompt) {
    const promptIds = tokenizer.encode(record.prompt, { bos: true });
    const completionIds = tokenizer.encode(record.completion, { eos: true });
    ids = [...promptIds, ...completionIds];
    promptLen = promptIds.length;
  } else {
    ids = tokenizer.encode(record.text, { bos: true, eos: true });
  }
  if (ids.length > maxSeqLen) {
    // Keep the prompt and truncate the tail; a truncated completion still
    // teaches the prefix, while dropping the prompt would teach nothing.
    ids = ids.slice(0, maxSeqLen);
  }
  return { id: record.id, ids, promptLen: Math.min(promptLen, ids.length - 1), chars: (record.text ?? '').length };
}

export function encodeRecords(records, tokenizer, options = {}) {
  const out = [];
  const skipped = [];
  for (const record of records) {
    const encoded = encodeRecord(record, tokenizer, options);
    if (encoded.ids.length < 2) {
      skipped.push({ id: record.id, reason: 'too_short_after_encoding' });
      continue;
    }
    out.push(encoded);
  }
  return { examples: out, skipped };
}

export function countTokens(examples) {
  return examples.reduce((n, e) => n + e.ids.length, 0);
}

// Deterministic batches: the rng is seeded per epoch by the caller.
export function makeBatches(examples, batchSize, rng) {
  const order = rng ? rng.shuffled(examples) : examples;
  const batches = [];
  for (let i = 0; i < order.length; i += batchSize) batches.push(order.slice(i, i + batchSize));
  return batches;
}
