import fs from 'node:fs';
import path from 'node:path';
import backend from '../src/train/backends/hf-peft.mjs';

const MODEL = path.resolve('.tmf-run/fixtures/tiny-llama');
const out = { steps: [] };
const step = (name, ok, detail) => { out.steps.push({ name, ok, detail }); console.log(`${ok ? 'ok' : 'FAIL'} ${name} ${JSON.stringify(detail)}`); };

const examples = fs.readFileSync('.tmf-run/fixtures/synth.jsonl', 'utf8').trim().split('\n').map((l) => JSON.parse(l)).slice(0, 8);

const base = { backend: 'hf-peft', kind: 'base', modelId: MODEL, dir: MODEL, adapterDir: null, mode: 'lora' };
const fp1 = backend.fingerprint(base);
const fp2 = backend.fingerprint(base);
step('fingerprint_stable', fp1 === fp2 && /^[0-9a-f]{64}$/.test(fp1), { fp: fp1.slice(0, 16) });

const fpMode = backend.fingerprint({ ...base, mode: 'full-finetune' });
step('fingerprint_mode_sensitive', fpMode !== fp1, { differs: fpMode !== fp1 });

// content sensitivity: perturb one byte of a copied weight dir
const tmpDir = path.resolve('.tmf-run/g12-fpdir');
fs.rmSync(tmpDir, { recursive: true, force: true });
fs.cpSync(MODEL, tmpDir, { recursive: true });
const fpCopy = backend.fingerprint({ ...base, modelId: MODEL, dir: tmpDir });
const wf = path.join(tmpDir, 'model.safetensors');
const buf = fs.readFileSync(wf); buf[buf.length - 1] ^= 0xff; fs.writeFileSync(wf, buf);
const fpMut = backend.fingerprint({ ...base, modelId: MODEL, dir: tmpDir });
step('fingerprint_content_sensitive', fpCopy !== fpMut && fpCopy === fp1, { sameAsOriginalBeforeMutation: fpCopy === fp1, changedAfterMutation: fpCopy !== fpMut });

// train via in-memory examples (uniform parameter contract, no examplesFile)
const outDir = path.resolve('.tmf-run/g12-train');
fs.rmSync(outDir, { recursive: true, force: true });
const t = await backend.train({
  handle: base, examples, mode: 'lora',
  hyper: { learningRate: 0.01, batchSize: 2, checkpointEvery: 8 },
  budget: { maxSteps: 8, maxSeconds: 120, maxMemoryMb: 4096 },
  outDir,
});
step('train_accepts_in_memory_examples', t.metrics.steps === 8 && t.metrics.parameterL2Delta > 0, {
  steps: t.metrics.steps, firstLoss: t.metrics.firstLoss, finalLoss: t.metrics.finalLoss, l2: t.metrics.parameterL2Delta,
  materialized: fs.existsSync(path.join(outDir, 'train-examples.jsonl')),
});

const fpTuned = backend.fingerprint(t.handle);
step('fingerprint_detects_adapter', fpTuned !== fp1, { base: fp1.slice(0, 12), tuned: fpTuned.slice(0, 12) });

let negOk = false; let negMsg = '';
try { await backend.train({ handle: base, mode: 'lora', outDir, budget: { maxSteps: 1 } }); } catch (e) { negOk = /requires examplesFile or examples/.test(e.message); negMsg = e.message; }
step('negative_no_examples', negOk, { message: negMsg });

out.failedSteps = out.steps.filter((s) => !s.ok).map((s) => s.name);
fs.writeFileSync('.tmf-run/artifacts/g12-verify.json', `${JSON.stringify(out, null, 2)}\n`);
console.log(`steps ${out.steps.length} failed ${out.failedSteps.length}`);
process.exit(out.failedSteps.length ? 1 : 0);
