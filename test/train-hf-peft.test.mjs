// Real-execution tests for the hf-peft backend.
//
// These tests run only when the python stack is importable AND the local
// random-init fixture exists (scripts/make-tiny-hf-fixture.py). Otherwise they
// skip with the concrete reason instead of asserting on a mock: an unverified
// capability must read as unverified.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import backend from '../src/train/backends/hf-peft.mjs';

const root = process.cwd();
const modelId = path.join(root, '.tmf-run/fixtures/tiny-llama');
const examplesFile = path.join(root, '.tmf-run/fixtures/synth.jsonl');
const runDir = path.join(root, '.tmf-run/test-hf-peft');
const budget = { maxSteps: 4, maxSeconds: 120, maxMemoryMb: 3072, threads: 1, checkpointEvery: 2 };

const probe = await backend.probe();
const fixtureReady = fs.existsSync(path.join(modelId, 'config.json')) && fs.existsSync(examplesFile);
const skip = !probe.available
  ? `hf-peft unavailable: ${probe.reason}`
  : !fixtureReady
    ? 'local fixture missing: run scripts/make-tiny-hf-fixture.py'
    : false;

test('hf-peft probe requires the full training dependency set', () => {
  // datasets/accelerate are used by the training path, so they must be probed.
  assert.ok(backend.capabilities.frameworks.includes('datasets'));
  if (!probe.available) {
    assert.match(probe.reason, /missing python packages|python interpreter unusable|not valid json/);
  }
});

test('hf-peft declares resume and cancel semantics, not bare booleans', () => {
  assert.equal(backend.capabilities.resumeGranularity, 'checkpoint-step');
  assert.equal(backend.capabilities.cancelSemantics, 'sigkill-worker');
  assert.equal(backend.capabilities.isolation, 'process-rlimit');
  for (const bound of ['max-memory-mb', 'max-cpu-seconds', 'max-steps']) {
    assert.ok(backend.capabilities.resourceBounds.includes(bound), bound);
  }
});

test('hf-peft trains, checkpoints, reloads, evaluates and generates for real', { skip }, async () => {
  const base = await backend.createModel({ modelId, outDir: runDir });
  const baseline = await backend.evaluate({ handle: base, examplesFile, budget, outDir: path.join(runDir, 'eval-base') });
  assert.ok(Number.isFinite(baseline.loss) && baseline.loss > 0);
  assert.equal(baseline.adapterLoaded, false);

  const trained = await backend.train({
    handle: base, examplesFile, mode: 'lora', outDir: path.join(runDir, 'lora'),
    hyper: { batchSize: 4, lr: 0.05, lora: { rank: 4, alpha: 8, targets: ['q_proj', 'v_proj'] } }, budget,
  });
  assert.ok(trained.history.length > 0, 'training must report a loss history');
  assert.ok(trained.metrics.parameterL2Delta > 0, 'weights must actually move');
  assert.ok(trained.metrics.trainableParams > 0);
  assert.ok(trained.checkpoints.length > 0, 'checkpointEvery must write checkpoints');

  const file = path.join(runDir, 'lora', 'tuned.ckpt.json');
  await backend.saveModel({ handle: trained.handle, file, kind: 'adapter' });
  const reloaded = await backend.loadModel({ file });
  const tuned = await backend.evaluate({ handle: reloaded, examplesFile, budget, outDir: path.join(runDir, 'eval-tuned') });
  assert.equal(tuned.adapterLoaded, true, 'reloaded adapter must be in the eval process');
  assert.ok(Number.isFinite(tuned.loss));

  const text = await backend.generate({ handle: reloaded, prompt: 'map alpha to', maxNewTokens: 3, budget });
  assert.equal(typeof text, 'string');
});

test('hf-peft enforces the memory bound instead of ignoring it', { skip }, async () => {
  const base = { backend: 'hf-peft', modelId, mode: 'lora' };
  await assert.rejects(
    () => backend.evaluate({ handle: base, examplesFile, budget: { ...budget, maxMemoryMb: 64 }, outDir: path.join(runDir, 'oom') }),
    (err) => ['BACKEND_MEMORY_LIMIT', 'BACKEND_RESOURCE_LIMIT', 'BACKEND_FAILED'].includes(err.code),
  );
});

test('hf-peft refuses a model that is not on local disk', { skip }, async () => {
  await assert.rejects(
    () => backend.generate({ handle: { modelId: path.join(root, '.tmf-run/fixtures/absent') }, prompt: 'x', maxNewTokens: 1, budget }),
    (err) => err.code === 'BACKEND_FAILED',
  );
});

// These run everywhere: fingerprinting is pure JS over files on disk, so it is
// part of the backend contract that must hold even where Python is absent.
test('hf-peft fingerprint digests weight content, not the directory path', async () => {
  const dir = path.join(runDir, 'fp', 'weights');
  const moved = path.join(runDir, 'fp', 'weights-moved');
  await fs.promises.mkdir(dir, { recursive: true });
  await fs.promises.writeFile(path.join(dir, 'adapter_model.safetensors'), 'weights-v1');
  await fs.promises.writeFile(path.join(dir, 'trainer_state.json.log'), 'ignored-by-extension');
  const handle = { backend: 'hf-peft', modelId: 'local/tiny', dir, mode: 'lora' };
  const first = backend.fingerprint(handle);
  assert.match(first, /^[0-9a-f]{64}$/);
  assert.equal(backend.fingerprint(handle), first);

  await fs.promises.cp(dir, moved, { recursive: true });
  assert.equal(backend.fingerprint({ ...handle, dir: moved }), first, 'moving the same weights must not change the fingerprint');

  await fs.promises.writeFile(path.join(moved, 'adapter_model.safetensors'), 'weights-v2');
  assert.notEqual(backend.fingerprint({ ...handle, dir: moved }), first, 'changed weights must change the fingerprint');
  assert.notEqual(backend.fingerprint({ ...handle, mode: 'full-finetune' }), first);
  assert.notEqual(backend.fingerprint({ ...handle, modelId: 'local/other' }), first);
  assert.throws(() => backend.fingerprint({}), /requires a handle with modelId/);
});

test('hf-peft train() rejects a call with neither examples nor examplesFile', async () => {
  await assert.rejects(
    () => backend.train({ handle: { modelId: 'local/tiny', mode: 'lora' }, outDir: path.join(runDir, 'no-examples'), budget: { maxSteps: 1 } }),
    (err) => /requires examplesFile or examples/.test(err.message) || err.code === 'BACKEND_UNAVAILABLE',
  );
});
