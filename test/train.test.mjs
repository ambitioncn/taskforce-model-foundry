import test from 'node:test';
import assert from 'node:assert/strict';
import { AdamW, lrSchedule, DEFAULT_OPTIMIZER_CONFIG } from '../src/train/optimizer.mjs';
import { buildTokenizer, encodeRecord, encodeRecords, countTokens, makeBatches } from '../src/train/dataset.mjs';
import {
  registerBackend, getBackend, listBackends, probeBackends,
  resolveBackend, clearBackends, loadDefaultBackends,
} from '../src/train/backends/index.mjs';
import { Rng } from '../src/core/rng.mjs';
import { MODES } from '../src/config.mjs';
import builtinBackend from '../src/train/backends/builtin-js.mjs';

function param(name, values) {
  const data = Float64Array.from(values);
  return {
    name,
    size: data.length,
    data,
    grad: new Float64Array(data.length),
    zeroGrad() { this.grad.fill(0); },
  };
}

// ---------------------------------------------------------------- optimizer

test('AdamW rejects unnamed parameters so state stays checkpointable', () => {
  assert.throws(() => new AdamW([{ size: 1, data: new Float64Array(1) }]), /needs a name/);
});

test('AdamW first step matches the closed-form AdamW update', () => {
  const p = param('w', [1]);
  const opt = new AdamW([p], { lr: 0.1, weightDecay: 0.01, clipNorm: 0 });
  p.grad[0] = 0.5;
  opt.applyStep();
  // step 1: bias correction makes mHat=g and vHat=g^2, so the ratio is sign(g).
  const cfg = { ...DEFAULT_OPTIMIZER_CONFIG, lr: 0.1, weightDecay: 0.01 };
  const decayed = 1 - cfg.lr * cfg.weightDecay * 1;
  const expected = decayed - cfg.lr * (0.5 / (Math.sqrt(0.25) + cfg.eps));
  assert.ok(Math.abs(p.data[0] - expected) < 1e-12, `${p.data[0]} vs ${expected}`);
  assert.equal(opt.step, 1);
});

test('AdamW clips by global gradient norm across all parameters', () => {
  const a = param('a', [0, 0]);
  const b = param('b', [0]);
  const opt = new AdamW([a, b], { lr: 0.1, clipNorm: 1 });
  a.grad.set([3, 4]);
  b.grad[0] = 0; // global norm is 5, well over clipNorm
  const info = opt.applyStep();
  assert.ok(Math.abs(info.gradNorm - 5) < 1e-12);
  assert.ok(Math.abs(info.clipScale - 1 / 5) < 1e-9);
});

test('AdamW with zero decay and no gradient leaves the weight untouched', () => {
  const p = param('w', [0.75]);
  const opt = new AdamW([p], { lr: 0.5, weightDecay: 0 });
  p.grad = null;
  opt.applyStep();
  assert.equal(p.data[0], 0.75);
});

test('AdamW minimises a convex quadratic', () => {
  const p = param('w', [3]);
  const opt = new AdamW([p], { lr: 0.1, weightDecay: 0, clipNorm: 0 });
  let loss = Infinity;
  for (let i = 0; i < 400; i += 1) {
    loss = p.data[0] * p.data[0]; // f(w)=w^2, f'(w)=2w
    p.grad[0] = 2 * p.data[0];
    opt.applyStep();
  }
  assert.ok(Math.abs(p.data[0]) < 0.05, `expected convergence toward 0, got ${p.data[0]}`);
  assert.ok(loss < 9);
});

test('AdamW state round-trips through stateDict/loadStateDict', () => {
  const p = param('w', [1, -1]);
  const opt = new AdamW([p], { lr: 0.05 });
  p.grad.set([0.3, -0.7]);
  opt.applyStep();
  const saved = JSON.parse(JSON.stringify(opt.stateDict()));

  // Resume from the same weights the save was taken at, so the only thing
  // under test is whether the optimizer moments were restored.
  const q = param('w', Array.from(p.data));
  const resumed = new AdamW([q], { lr: 0.05 }).loadStateDict(saved);
  assert.equal(resumed.step, opt.step);

  p.grad.set([0.1, 0.2]);
  q.grad.set([0.1, 0.2]);
  opt.applyStep();
  resumed.applyStep();
  assert.deepEqual(Array.from(q.data), Array.from(p.data));
});

test('loadStateDict refuses mismatched or missing parameter state', () => {
  const p = param('w', [1, 2]);
  const opt = new AdamW([p], {});
  assert.throws(() => opt.loadStateDict({ step: 1, params: {} }), /missing parameter "w"/);
  assert.throws(() => opt.loadStateDict({ step: 1, params: { w: { m: [0], v: [0] } } }), /expected 2/);
});

test('zeroGrad clears every parameter gradient', () => {
  const p = param('w', [1]);
  const opt = new AdamW([p], {});
  p.grad[0] = 9;
  opt.zeroGrad();
  assert.equal(p.grad[0], 0);
});

// ---------------------------------------------------------------- lr schedule

test('lrSchedule warms up linearly then decays by cosine to minRatio', () => {
  const total = 100;
  const opts = { warmupSteps: 10, minRatio: 0.1 };
  assert.ok(Math.abs(lrSchedule(0, total, opts) - 0.1) < 1e-12);
  assert.ok(Math.abs(lrSchedule(9, total, opts) - 1) < 1e-12);
  const start = lrSchedule(10, total, opts);
  const mid = lrSchedule(55, total, opts);
  const end = lrSchedule(total, total, opts);
  assert.ok(start > mid && mid > end, `${start} > ${mid} > ${end}`);
  assert.ok(Math.abs(end - 0.1) < 1e-9);
  for (let s = 10; s <= total; s += 1) {
    const v = lrSchedule(s, total, opts);
    assert.ok(v >= 0.1 - 1e-9 && v <= 1 + 1e-9, `multiplier out of range at step ${s}: ${v}`);
  }
});

test('lrSchedule is flat when totalSteps is unknown', () => {
  assert.equal(lrSchedule(5, 0), 1);
  assert.equal(lrSchedule(5, -1), 1);
});

// ---------------------------------------------------------------- dataset

const records = [
  { id: 'r1', prompt: 'route: disk full\n', completion: 'queue=infra\npriority=P1\n', text: 'route: disk full' },
  { id: 'r2', prompt: 'route: login fails\n', completion: 'queue=identity\npriority=P2\n', text: 'route: login fails' },
  { id: 'r3', prompt: null, completion: null, text: 'runbooks describe the escalation path' },
];

test('buildTokenizer covers prompts, completions and plain documents', () => {
  const tok = buildTokenizer(records, { maxVocab: 256 });
  for (const r of records) {
    const text = r.prompt !== null && r.prompt !== undefined ? `${r.prompt}${r.completion}` : r.text;
    assert.equal(tok.decode(tok.encode(text)), text);
  }
});

test('encodeRecord marks the prompt boundary so loss is taken on the completion', () => {
  const tok = buildTokenizer(records);
  const enc = encodeRecord(records[0], tok, { maxSeqLen: 256 });
  assert.ok(enc.promptLen > 0);
  assert.ok(enc.promptLen < enc.ids.length - 1);
  const prompt = tok.encode(records[0].prompt, { bos: true });
  assert.deepEqual(enc.ids.slice(0, prompt.length), prompt);
});

test('encodeRecord gives a document no prompt boundary', () => {
  const tok = buildTokenizer(records);
  const enc = encodeRecord(records[2], tok, { maxSeqLen: 256 });
  assert.equal(enc.promptLen, 0);
});

test('encodeRecord truncates to maxSeqLen and keeps promptLen in range', () => {
  const tok = buildTokenizer(records);
  const enc = encodeRecord(records[0], tok, { maxSeqLen: 6 });
  assert.equal(enc.ids.length, 6);
  assert.ok(enc.promptLen <= enc.ids.length - 1);
});

test('encodeRecords skips sequences too short to train on and reports why', () => {
  const tok = buildTokenizer(records);
  // maxSeqLen=1 leaves a single BOS token, which carries no next-token target.
  const { examples, skipped } = encodeRecords(records, tok, { maxSeqLen: 1 });
  assert.equal(examples.length, 0);
  assert.equal(skipped.length, 3);
  assert.deepEqual([...new Set(skipped.map((s) => s.reason))], ['too_short_after_encoding']);
  assert.deepEqual(skipped.map((s) => s.id), ['r1', 'r2', 'r3']);
});

// Documented current behaviour, not an endorsement: an empty document still
// encodes to BOS+EOS (length 2) and therefore survives the length filter, so
// empty records must be dropped by the R2 pipeline rather than here.
test('an empty document survives encoding as a BOS/EOS pair', () => {
  const tok = buildTokenizer(records);
  const { examples, skipped } = encodeRecords(
    [{ id: 'empty', prompt: null, completion: null, text: '' }],
    tok,
    { maxSeqLen: 64 },
  );
  assert.equal(skipped.length, 0);
  assert.equal(examples.length, 1);
  assert.equal(examples[0].ids.length, 2);
});

test('encodeRecords is deterministic and countTokens sums the sequences', () => {
  const tok = buildTokenizer(records);
  const a = encodeRecords(records, tok, { maxSeqLen: 64 });
  const b = encodeRecords(records, tok, { maxSeqLen: 64 });
  assert.deepEqual(a.examples, b.examples);
  assert.equal(countTokens(a.examples), a.examples.reduce((n, e) => n + e.ids.length, 0));
});

test('makeBatches partitions every example exactly once', () => {
  const examples = Array.from({ length: 7 }, (_, i) => ({ id: `e${i}`, ids: [1, 2] }));
  const batches = makeBatches(examples, 3, null);
  assert.deepEqual(batches.map((b) => b.length), [3, 3, 1]);
  assert.deepEqual(batches.flat().map((e) => e.id).sort(), examples.map((e) => e.id).sort());
});

test('makeBatches shuffles deterministically for a given seed', () => {
  const examples = Array.from({ length: 12 }, (_, i) => ({ id: `e${i}`, ids: [1, 2] }));
  const ids = (rng) => makeBatches(examples, 4, rng).flat().map((e) => e.id);
  const a = ids(new Rng('epoch-1'));
  const b = ids(new Rng('epoch-1'));
  const c = ids(new Rng('epoch-2'));
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
  assert.deepEqual([...a].sort(), examples.map((e) => e.id).sort());
});

// ---------------------------------------------------------------- backends

const stub = (id, probe) => ({
  id,
  capabilities: { modes: ['lora'], frameworks: ['stub'], devices: ['cpu'], supportsResume: false },
  probe,
  createModel() {}, loadModel() {}, saveModel() {}, train() {}, evaluate() {}, generate() {},
});

test('registerBackend enforces the backend interface', (t) => {
  t.after(() => clearBackends());
  clearBackends();
  assert.throws(() => registerBackend({}), /must expose an id/);
  assert.throws(() => registerBackend({ id: 'x', capabilities: { modes: ['lora'] } }), /missing required method probe/);
  const noModes = stub('y', () => ({ available: true }));
  noModes.capabilities = {};
  assert.throws(() => registerBackend(noModes), /capabilities\.modes/);
  registerBackend(stub('ok', () => ({ available: true })));
  assert.equal(getBackend('ok').id, 'ok');
  assert.deepEqual(listBackends().map((b) => b.id), ['ok']);
});

test('getBackend names the registered ids when one is unknown', (t) => {
  t.after(() => clearBackends());
  clearBackends();
  registerBackend(stub('ok', () => ({ available: true })));
  assert.throws(() => getBackend('nope'), /unknown trainer backend "nope".*ok/s);
});

test('probeBackends reports a throwing probe as unavailable instead of crashing', async (t) => {
  t.after(() => clearBackends());
  clearBackends();
  registerBackend(stub('good', () => ({ available: true })));
  registerBackend(stub('bad', () => { throw new Error('no torch here'); }));
  registerBackend(stub('absent', () => ({ available: false, reason: 'missing runtime' })));
  const probes = await probeBackends();
  const byId = Object.fromEntries(probes.map((p) => [p.id, p]));
  assert.equal(byId.good.available, true);
  assert.equal(byId.bad.available, false);
  assert.match(byId.bad.reason, /probe threw: no torch here/);
  assert.equal(byId.absent.reason, 'missing runtime');
  assert.ok(byId.good.capabilities.modes.includes('lora'));
});

test('resolveBackend refuses an unavailable backend with a typed error', async (t) => {
  t.after(() => clearBackends());
  clearBackends();
  registerBackend(stub('absent', () => ({ available: false, reason: 'missing runtime' })));
  registerBackend(stub('present', () => ({ available: true })));
  assert.equal((await resolveBackend('present')).id, 'present');
  await assert.rejects(() => resolveBackend('absent'), (err) => {
    assert.equal(err.code, 'BACKEND_UNAVAILABLE');
    assert.equal(err.backendProbe.available, false);
    return true;
  });
});

test('loadDefaultBackends registers the built-in and hf-peft backends and both probe without throwing', async (t) => {
  t.after(() => clearBackends());
  clearBackends();
  const listed = await loadDefaultBackends();
  const ids = listed.map((b) => b.id).sort();
  assert.deepEqual(ids, ['builtin-js', 'hf-peft']);
  const probes = await probeBackends();
  const builtin = probes.find((p) => p.id === 'builtin-js');
  const hf = probes.find((p) => p.id === 'hf-peft');
  assert.equal(builtin.available, true, 'the pure-JS backend must always be available');
  // hf-peft availability depends on the host python stack; either answer is a
  // recorded fact, but it must be a boolean with a reason when false.
  assert.equal(typeof hf.available, 'boolean');
  if (!hf.available) assert.ok(hf.reason && hf.reason.length > 0, 'an unavailable backend must say why');
});

// Regression: config accepted mode "full-finetune" while builtin-js did not
// advertise it, so the only always-available backend rejected a valid config
// at train time.
test('the builtin backend advertises every mode the config accepts', () => {
  for (const mode of MODES) {
    assert.ok(
      builtinBackend.capabilities.modes.includes(mode),
      `builtin-js must support config mode "${mode}"`,
    );
  }
});
