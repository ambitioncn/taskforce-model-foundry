// AdamW with decoupled weight decay and global-norm gradient clipping.
//
// State is kept per parameter name so an optimizer can be serialized into a job
// checkpoint and resumed without changing the trajectory.

export const DEFAULT_OPTIMIZER_CONFIG = {
  lr: 0.01,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 0.01,
  clipNorm: 1,
};

export class AdamW {
  constructor(params, config = {}) {
    this.config = { ...DEFAULT_OPTIMIZER_CONFIG, ...config };
    this.params = params;
    this.step = 0;
    this.state = new Map();
    for (const p of params) {
      if (!p.name) throw new Error('every trainable parameter needs a name so optimizer state can be checkpointed');
      this.state.set(p.name, { m: new Float64Array(p.size), v: new Float64Array(p.size) });
    }
  }

  gradNorm() {
    let total = 0;
    for (const p of this.params) {
      if (!p.grad) continue;
      for (let i = 0; i < p.grad.length; i += 1) total += p.grad[i] * p.grad[i];
    }
    return Math.sqrt(total);
  }

  applyStep({ lr } = {}) {
    const cfg = this.config;
    const learningRate = lr ?? cfg.lr;
    this.step += 1;

    let clipScale = 1;
    const norm = this.gradNorm();
    if (cfg.clipNorm && norm > cfg.clipNorm) clipScale = cfg.clipNorm / (norm + 1e-12);

    const bc1 = 1 - Math.pow(cfg.beta1, this.step);
    const bc2 = 1 - Math.pow(cfg.beta2, this.step);

    for (const p of this.params) {
      const st = this.state.get(p.name);
      const grad = p.grad;
      if (!grad) continue;
      for (let i = 0; i < p.size; i += 1) {
        const g = grad[i] * clipScale;
        st.m[i] = cfg.beta1 * st.m[i] + (1 - cfg.beta1) * g;
        st.v[i] = cfg.beta2 * st.v[i] + (1 - cfg.beta2) * g * g;
        const mHat = st.m[i] / bc1;
        const vHat = st.v[i] / bc2;
        // Decoupled decay: applied to the weight, not folded into the gradient.
        const decayed = p.data[i] - learningRate * cfg.weightDecay * p.data[i];
        p.data[i] = decayed - learningRate * (mHat / (Math.sqrt(vHat) + cfg.eps));
      }
    }
    return { gradNorm: norm, clipScale, lr: learningRate };
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }

  stateDict() {
    const out = { step: this.step, config: this.config, params: {} };
    for (const [name, st] of this.state) {
      out.params[name] = { m: Array.from(st.m), v: Array.from(st.v) };
    }
    return out;
  }

  loadStateDict(state) {
    this.step = state.step ?? 0;
    for (const [name, st] of this.state) {
      const saved = state.params?.[name];
      if (!saved) throw new Error(`optimizer state is missing parameter "${name}"`);
      if (saved.m.length !== st.m.length) {
        throw new Error(`optimizer state for "${name}" has length ${saved.m.length}, expected ${st.m.length}`);
      }
      st.m.set(saved.m);
      st.v.set(saved.v);
    }
    return this;
  }
}

// Linear warmup then cosine decay; returns the multiplier for the base lr.
export function lrSchedule(step, totalSteps, { warmupSteps = 0, minRatio = 0.1 } = {}) {
  if (totalSteps <= 0) return 1;
  if (warmupSteps > 0 && step < warmupSteps) return (step + 1) / warmupSteps;
  const progress = Math.min(1, Math.max(0, (step - warmupSteps) / Math.max(1, totalSteps - warmupSteps)));
  return minRatio + (1 - minRatio) * 0.5 * (1 + Math.cos(Math.PI * progress));
}
