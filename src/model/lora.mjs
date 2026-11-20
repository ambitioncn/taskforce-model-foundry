// LoRA adapters: W_eff = W + (A @ B) * (alpha / r).
//
// B is zero-initialized so a freshly attached adapter is mathematically a no-op.
// That property is what makes "the adapter actually changed the model" testable:
// before training, base and adapted outputs must match exactly.

import { matmul, randn, scale, zeros, add } from '../core/autograd.mjs';

export class LoraLayer {
  constructor(inDim, outDim, rank, alpha, rng, name) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.rank = rank;
    this.alpha = alpha;
    this.scaling = alpha / rank;
    this.A = randn(inDim, rank, rng, 1 / Math.sqrt(inDim), { requiresGrad: true, name: `${name}.lora_A` });
    this.B = zeros(rank, outDim, { requiresGrad: true, name: `${name}.lora_B` });
  }

  params() {
    return [this.A, this.B];
  }

  // delta(x) = (x @ A) @ B * scaling
  apply(x) {
    return scale(matmul(matmul(x, this.A), this.B), this.scaling);
  }

  forward(x, baseOut) {
    return add(baseOut, this.apply(x));
  }

  toJSON() {
    return {
      inDim: this.inDim,
      outDim: this.outDim,
      rank: this.rank,
      alpha: this.alpha,
      A: Array.from(this.A.data),
      B: Array.from(this.B.data),
    };
  }

  loadJSON(json) {
    if (json.inDim !== this.inDim || json.outDim !== this.outDim || json.rank !== this.rank) {
      throw new Error(`lora shape mismatch: expected ${this.inDim}x${this.rank}x${this.outDim}, got ${json.inDim}x${json.rank}x${json.outDim}`);
    }
    this.A.data.set(json.A);
    this.B.data.set(json.B);
    this.alpha = json.alpha;
    this.scaling = json.alpha / json.rank;
  }
}
