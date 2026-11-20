// Character-level tokenizer with explicit control tokens.
//
// A char vocab is the right trade-off here: the foundry trains genuinely small
// models on CPU, and a learned subword vocab would add a dependency without
// changing whether the weight update is real.

export const PAD = '<pad>';
export const BOS = '<bos>';
export const EOS = '<eos>';
export const UNK = '<unk>';
export const SEP = '<sep>';
export const SPECIALS = [PAD, BOS, EOS, UNK, SEP];

export class Tokenizer {
  constructor(tokens) {
    this.tokens = tokens.slice();
    this.index = new Map(this.tokens.map((t, i) => [t, i]));
    this.padId = this.index.get(PAD);
    this.bosId = this.index.get(BOS);
    this.eosId = this.index.get(EOS);
    this.unkId = this.index.get(UNK);
    this.sepId = this.index.get(SEP);
  }

  get size() {
    return this.tokens.length;
  }

  static fromTexts(texts, { maxVocab = 512 } = {}) {
    const counts = new Map();
    for (const text of texts) {
      for (const ch of String(text)) counts.set(ch, (counts.get(ch) ?? 0) + 1);
    }
    // Sort by frequency then codepoint so the vocab is deterministic.
    const chars = [...counts.entries()]
      .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1))
      .map(([ch]) => ch)
      .filter((ch) => !SPECIALS.includes(ch))
      .slice(0, Math.max(0, maxVocab - SPECIALS.length));
    return new Tokenizer([...SPECIALS, ...chars]);
  }

  static fromJSON(json) {
    return new Tokenizer(json.tokens);
  }

  toJSON() {
    return { kind: 'char-v1', tokens: this.tokens };
  }

  encode(text, { bos = false, eos = false } = {}) {
    const ids = [];
    if (bos) ids.push(this.bosId);
    for (const ch of String(text)) ids.push(this.index.has(ch) ? this.index.get(ch) : this.unkId);
    if (eos) ids.push(this.eosId);
    return ids;
  }

  decode(ids, { skipSpecial = true } = {}) {
    let out = '';
    for (const id of ids) {
      const tok = this.tokens[id];
      if (tok === undefined) continue;
      if (SPECIALS.includes(tok)) {
        if (skipSpecial) continue;
        out += tok;
        continue;
      }
      out += tok;
    }
    return out;
  }
}
