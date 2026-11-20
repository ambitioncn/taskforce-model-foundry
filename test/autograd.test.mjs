import test from 'node:test';
import assert from 'node:assert/strict';
import {
  tensor, zeros, filled, matmul, matmulT, add, addRow, mul, scale, gelu, layerNorm,
  causalSoftmax, sliceCols, concatCols, embeddingLookup, crossEntropy, sum, backward,
} from '../src/core/autograd.mjs';
import { Rng } from '../src/core/rng.mjs';

// Central-difference gradient check. Every op the transformer uses is verified
// against finite differences, so "the gradients are real" is a test result
// rather than a claim.
function gradCheck(buildLoss, params, { eps = 1e-3, tol = 2e-2 } = {}) {
  for (const p of params) p.zeroGrad();
  const loss = buildLoss();
  backward(loss);
  const analytic = params.map((p) => Float64Array.from(p.grad ?? new Float64Array(p.size)));
  const worst = { relative: 0, param: null, index: -1, analytic: 0, numeric: 0 };
  for (let pi = 0; pi < params.length; pi += 1) {
    const p = params[pi];
    for (let i = 0; i < p.size; i += 1) {
      const original = p.data[i];
      p.data[i] = original + eps;
      const plus = buildLoss().data[0];
      p.data[i] = original - eps;
      const minus = buildLoss().data[0];
      p.data[i] = original;
      const numeric = (plus - minus) / (2 * eps);
      const a = analytic[pi][i];
      const absolute = Math.abs(a - numeric);
      // Same float32 noise floor as the model-level check: gradients below it
      // are compared in absolute terms.
      const relative = absolute < 1e-4 ? 0 : absolute / Math.max(1e-9, Math.abs(a) + Math.abs(numeric));
      if (relative > worst.relative) {
        Object.assign(worst, { relative, param: p.name ?? `p${pi}`, index: i, analytic: a, numeric });
      }
    }
  }
  assert.ok(
    worst.relative < tol,
    `gradient mismatch at ${worst.param}[${worst.index}]: analytic=${worst.analytic} numeric=${worst.numeric} rel=${worst.relative}`,
  );
  return worst;
}

const rng = new Rng(11);
const rand = (rows, cols, name, requiresGrad = true) => {
  const t = zeros(rows, cols, { requiresGrad, name });
  for (let i = 0; i < t.size; i += 1) t.data[i] = rng.normal(0, 0.7);
  return t;
};

test('matmul gradients match finite differences', () => {
  const a = rand(3, 4, 'a');
  const b = rand(4, 2, 'b');
  gradCheck(() => sum(matmul(a, b)), [a, b]);
});

test('matmulT gradients match finite differences', () => {
  const a = rand(3, 4, 'a');
  const b = rand(5, 4, 'b');
  gradCheck(() => sum(mul(matmulT(a, b), matmulT(a, b))), [a, b]);
});

test('add, addRow, scale and mul gradients match finite differences', () => {
  const a = rand(3, 3, 'a');
  const b = rand(3, 3, 'b');
  const bias = rand(1, 3, 'bias');
  gradCheck(() => sum(mul(addRow(add(a, b), bias), scale(a, 0.5))), [a, b, bias]);
});

test('gelu gradients match finite differences', () => {
  const a = rand(4, 3, 'a');
  gradCheck(() => sum(mul(gelu(a), gelu(a))), [a]);
});

test('layerNorm gradients match finite differences', () => {
  const x = rand(4, 5, 'x');
  const gamma = filled(1, 5, 1.3, { requiresGrad: true, name: 'gamma' });
  const beta = rand(1, 5, 'beta');
  gradCheck(() => sum(mul(layerNorm(x, gamma, beta), layerNorm(x, gamma, beta))), [x, gamma, beta]);
});

test('causalSoftmax gradients match finite differences and respect the mask', () => {
  const scores = rand(5, 5, 'scores');
  const probs = causalSoftmax(scores);
  for (let i = 0; i < 5; i += 1) {
    let rowSum = 0;
    for (let j = 0; j < 5; j += 1) {
      if (j > i) assert.equal(probs.at(i, j), 0, `position ${i} must not attend to future position ${j}`);
      else rowSum += probs.at(i, j);
    }
    assert.ok(Math.abs(rowSum - 1) < 1e-5, `row ${i} should sum to 1, got ${rowSum}`);
  }
  const weights = rand(5, 5, 'w', false);
  gradCheck(() => sum(mul(causalSoftmax(scores), weights)), [scores]);
});

test('sliceCols and concatCols gradients match finite differences', () => {
  const a = rand(3, 6, 'a');
  gradCheck(() => {
    const left = sliceCols(a, 0, 3);
    const right = sliceCols(a, 3, 6);
    return sum(mul(concatCols([right, left]), concatCols([left, right])));
  }, [a]);
});

test('embeddingLookup accumulates gradients for repeated ids', () => {
  const table = rand(5, 3, 'table');
  const ids = [1, 3, 1, 1];
  gradCheck(() => sum(mul(embeddingLookup(table, ids), embeddingLookup(table, ids))), [table]);
  table.zeroGrad();
  backward(sum(embeddingLookup(table, ids)));
  // id 1 appears three times, id 3 once, ids 0/2/4 never.
  assert.ok(Math.abs(table.grad[1 * 3] - 3) < 1e-5);
  assert.ok(Math.abs(table.grad[3 * 3] - 1) < 1e-5);
  assert.equal(table.grad[0], 0);
  assert.equal(table.grad[4 * 3], 0);
});

test('crossEntropy gradients match finite differences and honour ignoreIndex', () => {
  const logits = rand(4, 6, 'logits');
  const targets = [2, 5, 0, 3];
  gradCheck(() => crossEntropy(logits, targets), [logits]);

  const masked = [-1, 5, -1, 3];
  logits.zeroGrad();
  const loss = crossEntropy(logits, masked, -1);
  assert.equal(loss.tokenCount, 2);
  backward(loss);
  for (let c = 0; c < 6; c += 1) {
    assert.equal(logits.grad[0 * 6 + c], 0, 'ignored positions must not produce gradient');
    assert.equal(logits.grad[2 * 6 + c], 0, 'ignored positions must not produce gradient');
  }
  assert.ok(Array.from(logits.grad.subarray(6, 12)).some((v) => v !== 0));
});

test('frozen leaves do not allocate gradients but still propagate upstream', () => {
  const frozen = tensor([[1, 2], [3, 4]], 2, 2, { requiresGrad: false, name: 'frozen' });
  const trainable = tensor([[1, 0], [0, 1]], 2, 2, { requiresGrad: true, name: 'trainable' });
  const loss = sum(matmul(frozen, trainable));
  backward(loss);
  assert.equal(frozen.grad, null, 'frozen leaf should not allocate a gradient buffer');
  assert.ok(trainable.grad.some((v) => v !== 0));
});
