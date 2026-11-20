// Minimal reverse-mode autodiff over 2D float32 tensors.
//
// Scope is deliberately narrow: the builtin trainer processes one sequence at a
// time, so every activation in the model is a [rows, cols] matrix. Keeping the
// engine 2D-only removes the stride bookkeeping a general tensor library needs
// while still supporting a real transformer with real gradients.

export class Tensor {
  constructor(data, rows, cols, { requiresGrad = false, name = null } = {}) {
    if (data.length !== rows * cols) {
      throw new Error(`tensor data length ${data.length} does not match shape ${rows}x${cols}`);
    }
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.rows = rows;
    this.cols = cols;
    this.requiresGrad = requiresGrad;
    this.name = name;
    this.grad = null;
    this.parents = [];
    this.backwardFn = null;
  }

  get size() {
    return this.rows * this.cols;
  }

  get isLeaf() {
    return this.parents.length === 0;
  }

  // Grads are materialized lazily: a frozen leaf never allocates one.
  ensureGrad() {
    if (!this.grad) this.grad = new Float32Array(this.size);
    return this.grad;
  }

  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }

  at(r, c) {
    return this.data[r * this.cols + c];
  }

  clone({ requiresGrad = this.requiresGrad, name = this.name } = {}) {
    return new Tensor(this.data.slice(), this.rows, this.cols, { requiresGrad, name });
  }

  toArray() {
    const out = [];
    for (let r = 0; r < this.rows; r += 1) out.push(Array.from(this.data.subarray(r * this.cols, (r + 1) * this.cols)));
    return out;
  }
}

export function tensor(values, rows, cols, opts) {
  const flat = Array.isArray(values[0]) ? values.flat() : values;
  const r = rows ?? (Array.isArray(values[0]) ? values.length : 1);
  const c = cols ?? (Array.isArray(values[0]) ? values[0].length : flat.length);
  return new Tensor(new Float32Array(flat), r, c, opts);
}

export function zeros(rows, cols, opts) {
  return new Tensor(new Float32Array(rows * cols), rows, cols, opts);
}

export function filled(rows, cols, value, opts) {
  const d = new Float32Array(rows * cols);
  d.fill(value);
  return new Tensor(d, rows, cols, opts);
}

export function randn(rows, cols, rng, std = 0.02, opts) {
  const d = new Float32Array(rows * cols);
  for (let i = 0; i < d.length; i += 1) d[i] = rng.normal(0, std);
  return new Tensor(d, rows, cols, opts);
}

// A tensor participates in the graph when it is trainable itself or descends
// from something trainable.
function tracked(...parents) {
  return parents.some((p) => p.requiresGrad || !p.isLeaf);
}

function makeNode(out, parents, backwardFn) {
  if (!tracked(...parents)) return out;
  out.parents = parents;
  out.backwardFn = backwardFn;
  return out;
}

function accumulate(t, fn) {
  // Frozen leaves (base weights during LoRA training) are dead ends: no grad
  // is needed by anything upstream, so skip the work entirely.
  if (t.isLeaf && !t.requiresGrad) return;
  fn(t.ensureGrad());
}

export function matmul(a, b) {
  if (a.cols !== b.rows) throw new Error(`matmul shape mismatch ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
  const m = a.rows;
  const k = a.cols;
  const n = b.cols;
  const out = zeros(m, n);
  const ad = a.data;
  const bd = b.data;
  const od = out.data;
  for (let i = 0; i < m; i += 1) {
    const aRow = i * k;
    const oRow = i * n;
    for (let p = 0; p < k; p += 1) {
      const av = ad[aRow + p];
      if (av === 0) continue;
      const bRow = p * n;
      for (let j = 0; j < n; j += 1) od[oRow + j] += av * bd[bRow + j];
    }
  }
  return makeNode(out, [a, b], (g) => {
    accumulate(a, (ga) => {
      for (let i = 0; i < m; i += 1) {
        for (let p = 0; p < k; p += 1) {
          let s = 0;
          const bRow = p * n;
          const gRow = i * n;
          for (let j = 0; j < n; j += 1) s += g[gRow + j] * bd[bRow + j];
          ga[i * k + p] += s;
        }
      }
    });
    accumulate(b, (gb) => {
      for (let p = 0; p < k; p += 1) {
        for (let j = 0; j < n; j += 1) {
          let s = 0;
          for (let i = 0; i < m; i += 1) s += ad[i * k + p] * g[i * n + j];
          gb[p * n + j] += s;
        }
      }
    });
  });
}

export function matmulT(a, b) {
  // a [m,k] @ b[n,k]^T -> [m,n]; avoids materializing the transpose in attention.
  if (a.cols !== b.cols) throw new Error(`matmulT shape mismatch ${a.rows}x${a.cols} @ (${b.rows}x${b.cols})^T`);
  const m = a.rows;
  const k = a.cols;
  const n = b.rows;
  const out = zeros(m, n);
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < m; i += 1) {
    for (let j = 0; j < n; j += 1) {
      let s = 0;
      for (let p = 0; p < k; p += 1) s += ad[i * k + p] * bd[j * k + p];
      out.data[i * n + j] = s;
    }
  }
  return makeNode(out, [a, b], (g) => {
    accumulate(a, (ga) => {
      for (let i = 0; i < m; i += 1) {
        for (let p = 0; p < k; p += 1) {
          let s = 0;
          for (let j = 0; j < n; j += 1) s += g[i * n + j] * bd[j * k + p];
          ga[i * k + p] += s;
        }
      }
    });
    accumulate(b, (gb) => {
      for (let j = 0; j < n; j += 1) {
        for (let p = 0; p < k; p += 1) {
          let s = 0;
          for (let i = 0; i < m; i += 1) s += g[i * n + j] * ad[i * k + p];
          gb[j * k + p] += s;
        }
      }
    });
  });
}

export function add(a, b) {
  if (a.rows !== b.rows || a.cols !== b.cols) throw new Error('add requires identical shapes');
  const out = zeros(a.rows, a.cols);
  for (let i = 0; i < out.size; i += 1) out.data[i] = a.data[i] + b.data[i];
  return makeNode(out, [a, b], (g) => {
    accumulate(a, (ga) => { for (let i = 0; i < g.length; i += 1) ga[i] += g[i]; });
    accumulate(b, (gb) => { for (let i = 0; i < g.length; i += 1) gb[i] += g[i]; });
  });
}

export function addRow(a, bias) {
  // bias is [1, cols], broadcast down the rows.
  if (bias.rows !== 1 || bias.cols !== a.cols) throw new Error('addRow requires a [1,cols] bias');
  const out = zeros(a.rows, a.cols);
  for (let r = 0; r < a.rows; r += 1) {
    for (let c = 0; c < a.cols; c += 1) out.data[r * a.cols + c] = a.data[r * a.cols + c] + bias.data[c];
  }
  return makeNode(out, [a, bias], (g) => {
    accumulate(a, (ga) => { for (let i = 0; i < g.length; i += 1) ga[i] += g[i]; });
    accumulate(bias, (gb) => {
      for (let r = 0; r < a.rows; r += 1) for (let c = 0; c < a.cols; c += 1) gb[c] += g[r * a.cols + c];
    });
  });
}

export function scale(a, factor) {
  const out = zeros(a.rows, a.cols);
  for (let i = 0; i < out.size; i += 1) out.data[i] = a.data[i] * factor;
  return makeNode(out, [a], (g) => {
    accumulate(a, (ga) => { for (let i = 0; i < g.length; i += 1) ga[i] += g[i] * factor; });
  });
}

export function mul(a, b) {
  if (a.rows !== b.rows || a.cols !== b.cols) throw new Error('mul requires identical shapes');
  const out = zeros(a.rows, a.cols);
  for (let i = 0; i < out.size; i += 1) out.data[i] = a.data[i] * b.data[i];
  return makeNode(out, [a, b], (g) => {
    accumulate(a, (ga) => { for (let i = 0; i < g.length; i += 1) ga[i] += g[i] * b.data[i]; });
    accumulate(b, (gb) => { for (let i = 0; i < g.length; i += 1) gb[i] += g[i] * a.data[i]; });
  });
}

const GELU_C = Math.sqrt(2 / Math.PI);

export function gelu(a) {
  const out = zeros(a.rows, a.cols);
  for (let i = 0; i < out.size; i += 1) {
    const x = a.data[i];
    out.data[i] = 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
  }
  return makeNode(out, [a], (g) => {
    accumulate(a, (ga) => {
      for (let i = 0; i < g.length; i += 1) {
        const x = a.data[i];
        const inner = GELU_C * (x + 0.044715 * x * x * x);
        const t = Math.tanh(inner);
        const dInner = GELU_C * (1 + 3 * 0.044715 * x * x);
        ga[i] += g[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner);
      }
    });
  });
}

export function layerNorm(x, gamma, beta, eps = 1e-5) {
  const { rows, cols } = x;
  const out = zeros(rows, cols);
  const cache = { mean: new Float64Array(rows), invStd: new Float64Array(rows), xhat: new Float64Array(rows * cols) };
  for (let r = 0; r < rows; r += 1) {
    let mean = 0;
    for (let c = 0; c < cols; c += 1) mean += x.data[r * cols + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c += 1) {
      const d = x.data[r * cols + c] - mean;
      variance += d * d;
    }
    variance /= cols;
    const invStd = 1 / Math.sqrt(variance + eps);
    cache.mean[r] = mean;
    cache.invStd[r] = invStd;
    for (let c = 0; c < cols; c += 1) {
      const xh = (x.data[r * cols + c] - mean) * invStd;
      cache.xhat[r * cols + c] = xh;
      out.data[r * cols + c] = xh * gamma.data[c] + beta.data[c];
    }
  }
  return makeNode(out, [x, gamma, beta], (g) => {
    accumulate(gamma, (gg) => {
      for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) gg[c] += g[r * cols + c] * cache.xhat[r * cols + c];
    });
    accumulate(beta, (gb) => {
      for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) gb[c] += g[r * cols + c];
    });
    accumulate(x, (gx) => {
      for (let r = 0; r < rows; r += 1) {
        let sumDxhat = 0;
        let sumDxhatXhat = 0;
        for (let c = 0; c < cols; c += 1) {
          const dxhat = g[r * cols + c] * gamma.data[c];
          sumDxhat += dxhat;
          sumDxhatXhat += dxhat * cache.xhat[r * cols + c];
        }
        for (let c = 0; c < cols; c += 1) {
          const dxhat = g[r * cols + c] * gamma.data[c];
          gx[r * cols + c] += cache.invStd[r] * (dxhat - sumDxhat / cols - cache.xhat[r * cols + c] * sumDxhatXhat / cols);
        }
      }
    });
  });
}

export function causalSoftmax(scores) {
  // scores is [T,T]; position i may only attend to j <= i.
  const { rows, cols } = scores;
  if (rows !== cols) throw new Error('causalSoftmax expects a square [T,T] score matrix');
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i += 1) {
    let max = -Infinity;
    for (let j = 0; j <= i; j += 1) max = Math.max(max, scores.data[i * cols + j]);
    let sum = 0;
    for (let j = 0; j <= i; j += 1) {
      const e = Math.exp(scores.data[i * cols + j] - max);
      out.data[i * cols + j] = e;
      sum += e;
    }
    for (let j = 0; j <= i; j += 1) out.data[i * cols + j] /= sum;
  }
  return makeNode(out, [scores], (g) => {
    accumulate(scores, (gs) => {
      for (let i = 0; i < rows; i += 1) {
        let dot = 0;
        for (let j = 0; j <= i; j += 1) dot += g[i * cols + j] * out.data[i * cols + j];
        for (let j = 0; j <= i; j += 1) {
          const p = out.data[i * cols + j];
          gs[i * cols + j] += p * (g[i * cols + j] - dot);
        }
      }
    });
  });
}

export function sliceCols(a, start, end) {
  const width = end - start;
  const out = zeros(a.rows, width);
  for (let r = 0; r < a.rows; r += 1) {
    for (let c = 0; c < width; c += 1) out.data[r * width + c] = a.data[r * a.cols + start + c];
  }
  return makeNode(out, [a], (g) => {
    accumulate(a, (ga) => {
      for (let r = 0; r < a.rows; r += 1) for (let c = 0; c < width; c += 1) ga[r * a.cols + start + c] += g[r * width + c];
    });
  });
}

export function concatCols(parts) {
  const rows = parts[0].rows;
  const cols = parts.reduce((n, p) => n + p.cols, 0);
  const out = zeros(rows, cols);
  let offset = 0;
  for (const part of parts) {
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < part.cols; c += 1) out.data[r * cols + offset + c] = part.data[r * part.cols + c];
    }
    offset += part.cols;
  }
  return makeNode(out, parts, (g) => {
    let off = 0;
    for (const part of parts) {
      const localOff = off;
      accumulate(part, (gp) => {
        for (let r = 0; r < rows; r += 1) {
          for (let c = 0; c < part.cols; c += 1) gp[r * part.cols + c] += g[r * cols + localOff + c];
        }
      });
      off += part.cols;
    }
  });
}

export function embeddingLookup(table, ids) {
  const out = zeros(ids.length, table.cols);
  for (let i = 0; i < ids.length; i += 1) {
    const id = ids[i];
    if (id < 0 || id >= table.rows) throw new Error(`embedding id ${id} out of range 0..${table.rows - 1}`);
    out.data.set(table.data.subarray(id * table.cols, (id + 1) * table.cols), i * table.cols);
  }
  return makeNode(out, [table], (g) => {
    accumulate(table, (gt) => {
      for (let i = 0; i < ids.length; i += 1) {
        const base = ids[i] * table.cols;
        for (let c = 0; c < table.cols; c += 1) gt[base + c] += g[i * table.cols + c];
      }
    });
  });
}

export function crossEntropy(logits, targets, ignoreIndex = -1) {
  // Mean token-level negative log likelihood over non-ignored positions.
  const { rows, cols } = logits;
  if (targets.length !== rows) throw new Error('crossEntropy needs one target per logit row');
  const probs = new Float64Array(rows * cols);
  let total = 0;
  let counted = 0;
  for (let r = 0; r < rows; r += 1) {
    if (targets[r] === ignoreIndex) continue;
    let max = -Infinity;
    for (let c = 0; c < cols; c += 1) max = Math.max(max, logits.data[r * cols + c]);
    let sum = 0;
    for (let c = 0; c < cols; c += 1) {
      const e = Math.exp(logits.data[r * cols + c] - max);
      probs[r * cols + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c += 1) probs[r * cols + c] /= sum;
    total += -Math.log(Math.max(probs[r * cols + targets[r]], 1e-12));
    counted += 1;
  }
  const loss = counted > 0 ? total / counted : 0;
  const out = new Tensor(new Float32Array([loss]), 1, 1);
  out.tokenCount = counted;
  // The float32 cell is what the graph carries; `lossValue` keeps the float64
  // accumulation so evaluation metrics and gradient checks are not limited by
  // the storage precision of a single scalar.
  out.lossValue = loss;
  return makeNode(out, [logits], (g) => {
    accumulate(logits, (gl) => {
      if (counted === 0) return;
      const upstream = g[0] / counted;
      for (let r = 0; r < rows; r += 1) {
        if (targets[r] === ignoreIndex) continue;
        for (let c = 0; c < cols; c += 1) {
          gl[r * cols + c] += upstream * (probs[r * cols + c] - (c === targets[r] ? 1 : 0));
        }
      }
    });
  });
}

export function sum(a) {
  let s = 0;
  for (let i = 0; i < a.size; i += 1) s += a.data[i];
  const out = new Tensor(new Float32Array([s]), 1, 1);
  return makeNode(out, [a], (g) => {
    accumulate(a, (ga) => { for (let i = 0; i < ga.length; i += 1) ga[i] += g[0]; });
  });
}

export function backward(root) {
  if (root.size !== 1) throw new Error('backward expects a scalar root');
  const order = [];
  const seen = new Set();
  const stack = [[root, false]];
  while (stack.length) {
    const frame = stack.pop();
    const [node, expanded] = frame;
    if (expanded) {
      order.push(node);
      continue;
    }
    if (seen.has(node)) continue;
    seen.add(node);
    stack.push([node, true]);
    for (const parent of node.parents) if (!seen.has(parent)) stack.push([parent, false]);
  }
  root.ensureGrad()[0] = 1;
  for (let i = order.length - 1; i >= 0; i -= 1) {
    const node = order[i];
    if (node.backwardFn && node.grad) node.backwardFn(node.grad);
  }
  return order.length;
}
