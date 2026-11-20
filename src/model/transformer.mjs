// A small decoder-only transformer with real forward and backward passes.
//
// Config defaults are sized so that CPU-only training on a synthetic corpus
// finishes in seconds, which is what lets the acceptance suite exercise a
// genuine weight update instead of asserting against a recorded number.

import {
  Tensor, zeros, filled, randn, matmul, matmulT, add, addRow, scale, gelu,
  layerNorm, causalSoftmax, sliceCols, concatCols, embeddingLookup, crossEntropy,
} from '../core/autograd.mjs';
import { Rng } from '../core/rng.mjs';
import { LoraLayer } from './lora.mjs';

export const DEFAULT_MODEL_CONFIG = {
  vocabSize: 96,
  dModel: 64,
  nLayers: 2,
  nHeads: 4,
  dFF: 128,
  maxSeqLen: 96,
  initStd: 0.02,
};

export const DEFAULT_LORA_CONFIG = {
  rank: 4,
  alpha: 8,
  targets: ['attn.q', 'attn.v'],
};

class Linear {
  constructor(inDim, outDim, rng, name, { bias = true, std } = {}) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.name = name;
    this.weight = randn(inDim, outDim, rng, std ?? 1 / Math.sqrt(inDim), { requiresGrad: true, name: `${name}.weight` });
    this.bias = bias ? zeros(1, outDim, { requiresGrad: true, name: `${name}.bias` }) : null;
    this.lora = null;
  }

  attachLora(rank, alpha, rng) {
    this.lora = new LoraLayer(this.inDim, this.outDim, rank, alpha, rng, this.name);
    return this.lora;
  }

  params() {
    return this.bias ? [this.weight, this.bias] : [this.weight];
  }

  forward(x) {
    let out = matmul(x, this.weight);
    if (this.bias) out = addRow(out, this.bias);
    if (this.lora) out = this.lora.forward(x, out);
    return out;
  }
}

class Block {
  constructor(cfg, rng, idx) {
    this.cfg = cfg;
    this.idx = idx;
    const d = cfg.dModel;
    this.ln1g = filled(1, d, 1, { requiresGrad: true, name: `block${idx}.ln1.gamma` });
    this.ln1b = zeros(1, d, { requiresGrad: true, name: `block${idx}.ln1.beta` });
    this.q = new Linear(d, d, rng, `block${idx}.attn.q`);
    this.k = new Linear(d, d, rng, `block${idx}.attn.k`);
    this.v = new Linear(d, d, rng, `block${idx}.attn.v`);
    this.o = new Linear(d, d, rng, `block${idx}.attn.o`);
    this.ln2g = filled(1, d, 1, { requiresGrad: true, name: `block${idx}.ln2.gamma` });
    this.ln2b = zeros(1, d, { requiresGrad: true, name: `block${idx}.ln2.beta` });
    this.ff1 = new Linear(d, cfg.dFF, rng, `block${idx}.ffn.in`);
    this.ff2 = new Linear(cfg.dFF, d, rng, `block${idx}.ffn.out`);
  }

  linears() {
    return { 'attn.q': this.q, 'attn.k': this.k, 'attn.v': this.v, 'attn.o': this.o, 'ffn.in': this.ff1, 'ffn.out': this.ff2 };
  }

  params() {
    return [
      this.ln1g, this.ln1b, ...this.q.params(), ...this.k.params(), ...this.v.params(), ...this.o.params(),
      this.ln2g, this.ln2b, ...this.ff1.params(), ...this.ff2.params(),
    ];
  }

  forward(x) {
    const h = layerNorm(x, this.ln1g, this.ln1b);
    const q = this.q.forward(h);
    const k = this.k.forward(h);
    const v = this.v.forward(h);
    const headDim = this.cfg.dModel / this.cfg.nHeads;
    const invSqrt = 1 / Math.sqrt(headDim);
    const heads = [];
    for (let head = 0; head < this.cfg.nHeads; head += 1) {
      const start = head * headDim;
      const end = start + headDim;
      const qh = sliceCols(q, start, end);
      const kh = sliceCols(k, start, end);
      const vh = sliceCols(v, start, end);
      const scores = scale(matmulT(qh, kh), invSqrt);
      heads.push(matmul(causalSoftmax(scores), vh));
    }
    const attn = this.o.forward(this.cfg.nHeads === 1 ? heads[0] : concatCols(heads));
    const res1 = add(x, attn);
    const h2 = layerNorm(res1, this.ln2g, this.ln2b);
    const ffn = this.ff2.forward(gelu(this.ff1.forward(h2)));
    return add(res1, ffn);
  }
}

export class TransformerModel {
  constructor(config = {}, { seed = 1 } = {}) {
    this.config = { ...DEFAULT_MODEL_CONFIG, ...config };
    if (this.config.dModel % this.config.nHeads !== 0) {
      throw new Error(`dModel ${this.config.dModel} must be divisible by nHeads ${this.config.nHeads}`);
    }
    this.seed = seed;
    const rng = new Rng(seed);
    const cfg = this.config;
    this.tokEmb = randn(cfg.vocabSize, cfg.dModel, rng, cfg.initStd, { requiresGrad: true, name: 'tok_emb' });
    this.posEmb = randn(cfg.maxSeqLen, cfg.dModel, rng, cfg.initStd, { requiresGrad: true, name: 'pos_emb' });
    this.blocks = Array.from({ length: cfg.nLayers }, (_, i) => new Block(cfg, rng, i));
    this.lnfg = filled(1, cfg.dModel, 1, { requiresGrad: true, name: 'ln_f.gamma' });
    this.lnfb = zeros(1, cfg.dModel, { requiresGrad: true, name: 'ln_f.beta' });
    this.head = new Linear(cfg.dModel, cfg.vocabSize, rng, 'lm_head');
    this.loraLayers = [];
  }

  baseParams() {
    return [
      this.tokEmb, this.posEmb, ...this.blocks.flatMap((b) => b.params()),
      this.lnfg, this.lnfb, ...this.head.params(),
    ];
  }

  loraParams() {
    return this.loraLayers.flatMap(({ layer }) => layer.params());
  }

  // Which parameters the optimizer may touch. During adapter post-training the
  // base stays frozen, which also stops the engine from allocating base grads.
  trainableParams({ mode = 'full' } = {}) {
    if (mode === 'lora') {
      if (this.loraLayers.length === 0) throw new Error('no LoRA adapter attached; call attachLora() first');
      return this.loraParams();
    }
    return this.baseParams();
  }

  setBaseFrozen(frozen) {
    for (const p of this.baseParams()) p.requiresGrad = !frozen;
  }

  attachLora(loraConfig = {}) {
    const cfg = { ...DEFAULT_LORA_CONFIG, ...loraConfig };
    const rng = new Rng(`${this.seed}:lora:${cfg.rank}:${cfg.alpha}`);
    this.loraLayers = [];
    for (const block of this.blocks) {
      const linears = block.linears();
      for (const target of cfg.targets) {
        const linear = linears[target];
        if (!linear) throw new Error(`unknown LoRA target "${target}"; valid: ${Object.keys(linears).join(', ')}`);
        const layer = linear.attachLora(cfg.rank, cfg.alpha, rng);
        this.loraLayers.push({ key: `block${block.idx}.${target}`, layer, linear });
      }
    }
    this.loraConfig = cfg;
    return this.loraLayers.length;
  }

  detachLora() {
    for (const { linear } of this.loraLayers) linear.lora = null;
    this.loraLayers = [];
    this.loraConfig = null;
  }

  get loraAttached() {
    return this.loraLayers.length > 0;
  }

  forward(ids) {
    if (ids.length === 0) throw new Error('forward requires at least one token');
    if (ids.length > this.config.maxSeqLen) {
      throw new Error(`sequence length ${ids.length} exceeds maxSeqLen ${this.config.maxSeqLen}`);
    }
    const tok = embeddingLookup(this.tokEmb, ids);
    const posIds = ids.map((_, i) => i);
    const pos = embeddingLookup(this.posEmb, posIds);
    let h = add(tok, pos);
    for (const block of this.blocks) h = block.forward(h);
    h = layerNorm(h, this.lnfg, this.lnfb);
    return this.head.forward(h);
  }

  // Next-token loss over a single sequence. `promptLen` masks prompt positions
  // so instruction-style data only trains on the completion.
  loss(ids, { promptLen = 0 } = {}) {
    if (ids.length < 2) throw new Error('loss requires at least two tokens');
    const inputs = ids.slice(0, -1);
    const targets = ids.slice(1);
    const masked = targets.map((t, i) => (i < Math.max(0, promptLen - 1) ? -1 : t));
    const logits = this.forward(inputs);
    return crossEntropy(logits, masked, -1);
  }

  zeroGrads(params) {
    for (const p of params ?? [...this.baseParams(), ...this.loraParams()]) p.zeroGrad();
  }

  parameterCount({ mode = 'full' } = {}) {
    return this.trainableParams({ mode }).reduce((n, p) => n + p.size, 0);
  }

  stateDict() {
    const out = {};
    for (const p of this.baseParams()) out[p.name] = { rows: p.rows, cols: p.cols, data: Array.from(p.data) };
    return out;
  }

  loadStateDict(state) {
    for (const p of this.baseParams()) {
      const entry = state[p.name];
      if (!entry) throw new Error(`checkpoint is missing parameter "${p.name}"`);
      if (entry.rows !== p.rows || entry.cols !== p.cols) {
        throw new Error(`parameter "${p.name}" shape mismatch: checkpoint ${entry.rows}x${entry.cols} vs model ${p.rows}x${p.cols}`);
      }
      p.data.set(entry.data);
    }
    return this;
  }

  adapterState() {
    if (!this.loraAttached) throw new Error('no adapter attached');
    const layers = {};
    for (const { key, layer } of this.loraLayers) layers[key] = layer.toJSON();
    return { kind: 'lora-v1', config: this.loraConfig, layers };
  }

  loadAdapterState(state) {
    if (!this.loraAttached) this.attachLora(state.config);
    for (const { key, layer } of this.loraLayers) {
      const entry = state.layers[key];
      if (!entry) throw new Error(`adapter is missing layer "${key}"`);
      layer.loadJSON(entry);
    }
    return this;
  }
}

export function paramVectorDigestInput(params) {
  // Flattens parameters in a stable order for checksum / drift comparison.
  const out = [];
  for (const p of params) out.push(p.name, p.rows, p.cols, ...p.data);
  return out;
}

export function l2Delta(before, after) {
  let sum = 0;
  for (const name of Object.keys(before)) {
    const a = before[name];
    const b = after[name];
    if (!b) continue;
    for (let i = 0; i < a.length; i += 1) {
      const d = a[i] - b[i];
      sum += d * d;
    }
  }
  return Math.sqrt(sum);
}

export function snapshotParams(params) {
  const out = {};
  for (const p of params) out[p.name] = Float64Array.from(p.data);
  return out;
}

export { Tensor };
