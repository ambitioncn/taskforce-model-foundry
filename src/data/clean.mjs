// Normalization and quality filtering.

// Built with new RegExp so the source stays plain ASCII.
const CONTROL_CHARS = new RegExp('[\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f\\u007f]', 'g');
// Non-global twin: `.test` on a global regex would advance lastIndex.
const NON_PRINTABLE = new RegExp('[\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f\\u007f]');
const ZERO_WIDTH = new RegExp('[\\u200b-\\u200f\\u2028\\u2029\\u2060\\ufeff]', 'g');

export const DEFAULT_CLEAN_CONFIG = {
  normalizeUnicode: 'NFKC',
  collapseWhitespace: true,
  stripControlChars: true,
  trim: true,
  lowercase: false,
};

export function cleanText(text, config = {}) {
  const cfg = { ...DEFAULT_CLEAN_CONFIG, ...config };
  let value = String(text ?? '');
  const applied = [];
  if (cfg.normalizeUnicode) {
    const normalized = value.normalize(cfg.normalizeUnicode);
    if (normalized !== value) applied.push('unicode');
    value = normalized;
  }
  if (cfg.stripControlChars) {
    CONTROL_CHARS.lastIndex = 0;
    ZERO_WIDTH.lastIndex = 0;
    const stripped = value.replace(CONTROL_CHARS, '').replace(ZERO_WIDTH, '');
    if (stripped !== value) applied.push('control_chars');
    value = stripped;
  }
  if (cfg.collapseWhitespace) {
    // Preserve paragraph structure but drop runs of spaces/tabs and 3+ newlines.
    const collapsed = value.replace(/[ \t]+/g, ' ').replace(/ ?\n ?/g, '\n').replace(/\n{3,}/g, '\n\n');
    if (collapsed !== value) applied.push('whitespace');
    value = collapsed;
  }
  if (cfg.lowercase) {
    const lowered = value.toLowerCase();
    if (lowered !== value) applied.push('lowercase');
    value = lowered;
  }
  if (cfg.trim) {
    const trimmed = value.trim();
    if (trimmed !== value) applied.push('trim');
    value = trimmed;
  }
  return { text: value, applied };
}

// Normalized key for exact-duplicate and leakage comparison. Deliberately more
// aggressive than cleanText: case and punctuation are flattened so cosmetic
// edits cannot smuggle a holdout sample into the training set.
export function canonicalKey(text) {
  return String(text ?? '')
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

export const DEFAULT_QUALITY_CONFIG = {
  minChars: 8,
  maxChars: 4000,
  maxRepeatRatio: 0.35,
  minAlnumRatio: 0.45,
  maxNonPrintableRatio: 0.01,
  rejectPlaceholders: ['lorem ipsum', 'todo', 'tbd', 'xxxxx'],
};

export function qualityReport(text, config = {}) {
  const cfg = { ...DEFAULT_QUALITY_CONFIG, ...config };
  const value = String(text ?? '');
  const chars = [...value];
  const reasons = [];

  if (chars.length < cfg.minChars) reasons.push(`too_short(${chars.length}<${cfg.minChars})`);
  if (chars.length > cfg.maxChars) reasons.push(`too_long(${chars.length}>${cfg.maxChars})`);

  const alnum = chars.filter((c) => /[\p{L}\p{N}]/u.test(c)).length;
  const alnumRatio = chars.length ? alnum / chars.length : 0;
  if (chars.length >= cfg.minChars && alnumRatio < cfg.minAlnumRatio) {
    reasons.push(`low_alnum_ratio(${alnumRatio.toFixed(2)}<${cfg.minAlnumRatio})`);
  }

  const nonPrintable = chars.filter((c) => NON_PRINTABLE.test(c)).length;
  const nonPrintableRatio = chars.length ? nonPrintable / chars.length : 0;
  if (nonPrintableRatio > cfg.maxNonPrintableRatio) {
    reasons.push(`non_printable(${nonPrintableRatio.toFixed(3)}>${cfg.maxNonPrintableRatio})`);
  }

  // Repetition: share of the text taken by its most common 4-gram.
  let repeatRatio = 0;
  if (chars.length >= 16) {
    const counts = new Map();
    for (let i = 0; i + 4 <= chars.length; i += 1) {
      const gram = chars.slice(i, i + 4).join('');
      counts.set(gram, (counts.get(gram) ?? 0) + 1);
    }
    const total = chars.length - 3;
    const max = Math.max(...counts.values());
    repeatRatio = max / total;
    if (repeatRatio > cfg.maxRepeatRatio) reasons.push(`repetitive(${repeatRatio.toFixed(2)}>${cfg.maxRepeatRatio})`);
  }

  const lower = value.toLowerCase();
  const placeholder = cfg.rejectPlaceholders.find((p) => lower.includes(p));
  if (placeholder) reasons.push(`placeholder(${placeholder})`);

  const score = Number(
    Math.max(0, Math.min(1, 0.5 * alnumRatio + 0.5 * (1 - Math.min(1, repeatRatio / Math.max(cfg.maxRepeatRatio, 1e-6))))).toFixed(4),
  );
  return { ok: reasons.length === 0, reasons, score, metrics: { chars: chars.length, alnumRatio, repeatRatio, nonPrintableRatio } };
}
