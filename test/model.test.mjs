import test from 'node:test';
import assert from 'node:assert/strict';
import { backward } from '../src/core/autograd.mjs';
import { TransformerModel, snapshotParams, l2Delta } from '../src/model/transformer.mjs';
import { Tokenizer, SPECIALS } from '../src/model/tokenizer.mjs';

const tiny = { vocabSize: 12, dModel: 8, nLayers: 2, nHeads: 2, dFF: 16, maxSeqLen: 12 };

test('model construction is deterministic for a given seed', () => {
  const a = new TransformerModel(tiny, { seed: 5 });
  const b = new TransformerModel(tiny, { seed: 5 });
  const c = new TransformerModel(tiny, { seed: 6 });
  assert.deepEqual(Array.from(a.tokEmb.data), Array.from(b.tokEmb.data));
  assert.notDeepEqual(Array.from(a.tokEmb.data), Array.from(c.tokEmb.data));
});

test('model rejects an invalid head/width combination', () => {
  assert.throws(() => new TransformerModel({ ...tiny, dModel: 9, nHeads: 2 }), /divisible/);
});

test('end-to-end model gradients match finite differences', () => {
  const model = new TransformerModel(tiny, { seed: 7 });
  const ids = [1, 4, 5, 6, 2];
  // Sample across parameter kinds rather than all 1.4k entries: the per-op
  // checks already cover the ops, this confirms the assembled graph.
  const probes = [
    [model.tokEmb, 4 * tiny.dModel + 1],
    [model.posEmb, 2 * tiny.dModel + 3],
    [model.blocks[0].q.weight, 5],
    [model.blocks[0].k.weight, 11],
    [model.blocks[0].v.bias, 2],
    [model.blocks[0].ln1g, 3],
    [model.blocks[1].ff1.weight, 9],
    [model.blocks[1].ff2.bias, 1],
    [model.lnfb, 0],
    [model.head.weight, 17],
  ];
  model.zeroGrads();
  backward(model.loss(ids));
  const eps = 1e-3;
  // Activations are float32, so a central difference over a loss of order 1
  // carries roughly 1e-4 of quantisation noise. Parameters whose true gradient
  // is near that floor are judged on absolute error instead of relative.
  const NOISE_FLOOR = 1e-4;
  for (const [param, index] of probes) {
    const analytic = param.grad[index];
    const original = param.data[index];
    param.data[index] = original + eps;
    const plus = model.loss(ids).lossValue;
    param.data[index] = original - eps;
    const minus = model.loss(ids).lossValue;
    param.data[index] = original;
    const numeric = (plus - minus) / (2 * eps);
    const absolute = Math.abs(analytic - numeric);
    const rel = absolute / Math.max(1e-9, Math.abs(analytic) + Math.abs(numeric));
    assert.ok(
      rel < 3e-2 || absolute < NOISE_FLOOR,
      `${param.name}[${index}]: analytic=${analytic} numeric=${numeric} abs=${absolute} rel=${rel}`,
    );
  }
});

test('prompt masking excludes prompt positions from the loss', () => {
  const model = new TransformerModel(tiny, { seed: 8 });
  const ids = [1, 3, 4, 5, 6, 7];
  const full = model.loss(ids);
  const masked = model.loss(ids, { promptLen: 3 });
  assert.equal(full.tokenCount, 5);
  assert.equal(masked.tokenCount, 3);
  assert.notEqual(full.data[0], masked.data[0]);
});

test('a freshly attached LoRA adapter is a mathematical no-op', () => {
  const model = new TransformerModel(tiny, { seed: 9 });
  const ids = [1, 2, 3, 4];
  const before = model.forward(ids.slice(0, -1)).toArray();
  const count = model.attachLora({ rank: 2, alpha: 4 });
  assert.equal(count, tiny.nLayers * 2);
  const after = model.forward(ids.slice(0, -1)).toArray();
  assert.deepEqual(after, before, 'zero-initialised B must leave the base output untouched');
});

test('LoRA training mode freezes base weights and only exposes adapter params', () => {
  const model = new TransformerModel(tiny, { seed: 10 });
  model.attachLora({ rank: 2, alpha: 4 });
  model.setBaseFrozen(true);
  const trainable = model.trainableParams({ mode: 'lora' });
  assert.ok(trainable.length > 0);
  assert.ok(trainable.every((p) => p.name.includes('lora_')));
  const baseSnapshot = snapshotParams(model.baseParams());
  model.zeroGrads();
  backward(model.loss([1, 2, 3, 4, 5]));
  for (const p of model.baseParams()) {
    assert.ok(!p.grad || p.grad.every((v) => v === 0), `${p.name} must stay gradient-free while frozen`);
  }
  assert.ok(trainable.some((p) => p.grad && p.grad.some((v) => v !== 0)), 'adapter params must receive gradient');
  assert.equal(l2Delta(baseSnapshot, snapshotParams(model.baseParams())), 0);
});

test('trainableParams rejects lora mode before an adapter is attached', () => {
  const model = new TransformerModel(tiny, { seed: 11 });
  assert.throws(() => model.trainableParams({ mode: 'lora' }), /no LoRA adapter attached/);
});

test('attachLora rejects an unknown target', () => {
  const model = new TransformerModel(tiny, { seed: 12 });
  assert.throws(() => model.attachLora({ targets: ['attn.nope'] }), /unknown LoRA target/);
});

test('state dict round-trips and detects shape drift', () => {
  const a = new TransformerModel(tiny, { seed: 13 });
  const b = new TransformerModel(tiny, { seed: 14 });
  const ids = [1, 2, 3, 4];
  assert.notDeepEqual(b.forward(ids).toArray(), a.forward(ids).toArray());
  b.loadStateDict(a.stateDict());
  assert.deepEqual(b.forward(ids).toArray(), a.forward(ids).toArray());

  const wider = new TransformerModel({ ...tiny, dFF: 24 }, { seed: 13 });
  assert.throws(() => wider.loadStateDict(a.stateDict()), /shape mismatch/);
  const partial = a.stateDict();
  delete partial.tok_emb;
  assert.throws(() => new TransformerModel(tiny, { seed: 1 }).loadStateDict(partial), /missing parameter/);
});

test('adapter state round-trips onto a clean model', () => {
  const a = new TransformerModel(tiny, { seed: 15 });
  a.attachLora({ rank: 2, alpha: 4 });
  // Perturb the adapter so it is no longer the identity.
  for (const { layer } of a.loraLayers) for (let i = 0; i < layer.B.size; i += 1) layer.B.data[i] = 0.05 * (i + 1);
  const ids = [1, 2, 3, 4];
  const expected = a.forward(ids).toArray();

  const b = new TransformerModel(tiny, { seed: 15 });
  b.loadAdapterState(a.adapterState());
  assert.deepEqual(b.forward(ids).toArray(), expected, 'reloaded adapter must reproduce outputs exactly');

  const mismatched = JSON.parse(JSON.stringify(a.adapterState()));
  mismatched.layers['block0.attn.q'].rank = 3;
  const c = new TransformerModel(tiny, { seed: 15 });
  c.attachLora({ rank: 2, alpha: 4 });
  assert.throws(() => c.loadAdapterState(mismatched), /lora shape mismatch/);
});

test('forward rejects sequences beyond maxSeqLen and empty input', () => {
  const model = new TransformerModel(tiny, { seed: 16 });
  assert.throws(() => model.forward([]), /at least one token/);
  assert.throws(() => model.forward(new Array(tiny.maxSeqLen + 1).fill(1)), /exceeds maxSeqLen/);
  assert.throws(() => model.loss([1]), /at least two tokens/);
  assert.throws(() => model.forward([999]), /out of range/);
});

test('tokenizer builds a deterministic vocab and round-trips text', () => {
  const texts = ['hello foundry', 'hello hermes'];
  const t1 = Tokenizer.fromTexts(texts);
  const t2 = Tokenizer.fromTexts(texts);
  assert.deepEqual(t1.tokens, t2.tokens);
  assert.deepEqual(t1.tokens.slice(0, SPECIALS.length), SPECIALS);
  const ids = t1.encode('hello', { bos: true, eos: true });
  assert.equal(ids[0], t1.bosId);
  assert.equal(ids[ids.length - 1], t1.eosId);
  assert.equal(t1.decode(ids), 'hello');
  assert.equal(t1.decode(t1.encode('zzz')), '', 'unknown chars map to <unk> and drop on decode');
  assert.deepEqual(Tokenizer.fromJSON(t1.toJSON()).tokens, t1.tokens);
  assert.ok(Tokenizer.fromTexts(texts, { maxVocab: 8 }).size <= 8);
});
