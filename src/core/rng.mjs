// Deterministic, seedable PRNG. Every stochastic step in the foundry draws from
// one of these so that a (seed, config) pair reproduces a run bit-for-bit.

const MASK = 0xffffffff;

export function hashSeed(value) {
  // FNV-1a over the string form; keeps string seeds usable in configs.
  let h = 0x811c9dc5;
  const s = String(value);
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) & MASK;
  }
  return h >>> 0;
}

export class Rng {
  constructor(seed = 1) {
    const s = typeof seed === 'number' && Number.isFinite(seed) ? seed >>> 0 : hashSeed(seed);
    // mulberry32 state; 0 is a degenerate state so bump it.
    this.state = (s || 0x9e3779b9) >>> 0;
    this.seed = s;
  }

  next() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  int(maxExclusive) {
    return Math.floor(this.next() * maxExclusive);
  }

  pick(list) {
    return list[this.int(list.length)];
  }

  // Box-Muller; cached second draw keeps the stream aligned.
  normal(mean = 0, std = 1) {
    if (this._spare !== undefined) {
      const v = this._spare;
      this._spare = undefined;
      return mean + std * v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    this._spare = r * Math.sin(theta);
    return mean + std * r * Math.cos(theta);
  }

  shuffled(list) {
    const out = list.slice();
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = this.int(i + 1);
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  fork(label) {
    return new Rng(hashSeed(`${this.seed}:${label}`));
  }
}
