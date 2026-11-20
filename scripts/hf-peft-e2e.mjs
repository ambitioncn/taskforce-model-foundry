// Bounded real end-to-end exercise of the hf-peft backend on CPU.
//
// Uses only the locally generated random-init fixture and synthetic data from
// scripts/make-tiny-hf-fixture.py: no downloads, no accounts, no paid compute.
// Writes a JSON report so the result is auditable rather than narrated.
import path from 'node:path';
import fs from 'node:fs/promises';
import backend from '../src/train/backends/hf-peft.mjs';

const root = process.cwd();
const modelId = path.join(root, '.tmf-run/fixtures/tiny-llama');
const examplesFile = path.join(root, '.tmf-run/fixtures/synth.jsonl');
const runDir = path.join(root, '.tmf-run/e2e');
const report = { startedAt: new Date().toISOString(), steps: {} };

const record = async (name, fn) => {
  const t0 = Date.now();
  try {
    const value = await fn();
    report.steps[name] = { ok: true, ms: Date.now() - t0, value };
    console.log(`[ok] ${name}`);
    return value;
  } catch (err) {
    report.steps[name] = { ok: false, ms: Date.now() - t0, error: err.message.slice(0, 400), code: err.code };
    console.log(`[fail] ${name}: ${err.code ?? ''} ${err.message.slice(0, 200)}`);
    return null;
  }
};

const budget = { maxSteps: 24, checkpointEvery: 8, maxSeconds: 180, maxMemoryMb: 3072, threads: 1 };

const probe = await record('probe', () => backend.probe());
if (!probe?.available) {
  report.conclusion = 'backend unavailable; nothing else attempted';
  await fs.writeFile('.tmf-run/artifacts/hf-peft-e2e.json', JSON.stringify(report, null, 2));
  process.exit(1);
}

const base = await record('createModel', () => backend.createModel({ modelId, outDir: runDir }));

// 1. Independent baseline evaluation of the untrained base model.
const baseline = await record('evaluate_baseline', () => backend.evaluate({
  handle: base, examplesFile, budget, outDir: path.join(runDir, 'eval-baseline'),
}));

// 2. LoRA training with step checkpoints.
const trained = await record('train_lora', () => backend.train({
  handle: base, examplesFile, mode: 'lora', outDir: path.join(runDir, 'lora'),
  hyper: { batchSize: 4, lr: 0.05, lora: { rank: 4, alpha: 8, targets: ['q_proj', 'v_proj'] } },
  budget,
}));

// 3. Checkpoint save + reload through the backend's own serialization.
const ckptFile = path.join(runDir, 'lora', 'tuned.ckpt.json');
await record('saveModel', () => backend.saveModel({
  handle: trained?.handle ?? base, file: ckptFile, kind: 'adapter', meta: { steps: trained?.metrics?.totalSteps },
}));
const reloaded = await record('loadModel', () => backend.loadModel({ file: ckptFile }));

// 4. Independent evaluation of the reloaded tuned adapter (fresh process).
const tuned = await record('evaluate_tuned_reloaded', () => backend.evaluate({
  handle: reloaded, examplesFile, budget, outDir: path.join(runDir, 'eval-tuned'),
}));

// 5. Generation from the reloaded tuned model.
await record('generate_tuned', () => backend.generate({
  handle: reloaded, prompt: 'map alpha to', maxNewTokens: 4, temperature: 0, budget,
}));
await record('generate_baseline', () => backend.generate({
  handle: base, prompt: 'map alpha to', maxNewTokens: 4, temperature: 0, budget,
}));

// 6. Resume from the last written Trainer checkpoint.
const lastCkpt = (trained?.checkpoints ?? []).slice(-1)[0];
await record('train_resume_from_checkpoint', () => {
  if (!lastCkpt) throw new Error('no trainer checkpoint directory was written');
  return backend.train({
    handle: base, examplesFile, mode: 'lora', outDir: path.join(runDir, 'lora'),
    hyper: { batchSize: 4, lr: 0.05, lora: { rank: 4, alpha: 8, targets: ['q_proj', 'v_proj'] } },
    budget: { ...budget, maxSteps: 32 }, resume: lastCkpt,
  });
});

// 7. Full fine-tune mode actually trains all weights.
await record('train_full_finetune', () => backend.train({
  handle: base, examplesFile, mode: 'full-finetune', outDir: path.join(runDir, 'full'),
  hyper: { batchSize: 4, lr: 0.01 }, budget: { ...budget, maxSteps: 6, checkpointEvery: 0 },
}));

// 8. Negative: memory bound must terminate the worker, not be ignored.
await record('negative_memory_limit', async () => {
  try {
    await backend.evaluate({
      handle: base, examplesFile, budget: { ...budget, maxMemoryMb: 64 },
      outDir: path.join(runDir, 'eval-oom'),
    });
    throw new Error('UNEXPECTED: evaluate succeeded under a 64MB address-space limit');
  } catch (err) {
    if (err.message.startsWith('UNEXPECTED')) throw err;
    return { rejectedWith: err.code, detail: (err.stderrTail ?? err.message).slice(-200) };
  }
});

// 9. Negative: cancellation must kill the worker.
await record('negative_cancel', async () => {
  const ac = new AbortController();
  const p = backend.train({
    handle: base, examplesFile, mode: 'lora', outDir: path.join(runDir, 'cancel'),
    hyper: { batchSize: 4, lr: 0.05 }, budget: { ...budget, maxSteps: 2000, checkpointEvery: 0 },
    signal: ac.signal,
  });
  setTimeout(() => ac.abort(), 4000);
  try {
    await p;
    throw new Error('UNEXPECTED: cancelled training resolved');
  } catch (err) {
    if (err.message.startsWith('UNEXPECTED')) throw err;
    return { rejectedWith: err.code, message: err.message.slice(0, 160) };
  }
});

// 10. Negative: missing local model must fail offline, never fetch.
await record('negative_missing_local_model', async () => {
  try {
    await backend.generate({
      handle: { modelId: path.join(root, '.tmf-run/fixtures/does-not-exist') },
      prompt: 'map alpha to', maxNewTokens: 2, budget,
    });
    throw new Error('UNEXPECTED: generate succeeded for a missing local model');
  } catch (err) {
    if (err.message.startsWith('UNEXPECTED')) throw err;
    return { rejectedWith: err.code, detail: (err.stderrTail ?? err.message).slice(-160) };
  }
});

report.summary = {
  baselineLoss: baseline?.loss ?? null,
  tunedLoss: tuned?.loss ?? null,
  lossImproved: baseline && tuned ? tuned.loss < baseline.loss : null,
  trainFirstLoss: trained?.metrics?.firstLoss ?? null,
  trainFinalLoss: trained?.metrics?.finalLoss ?? null,
  parameterL2Delta: trained?.metrics?.parameterL2Delta ?? null,
  trainableParams: trained?.metrics?.trainableParams ?? null,
  checkpointsWritten: (trained?.checkpoints ?? []).length,
  failedSteps: Object.entries(report.steps).filter(([, v]) => !v.ok).map(([k]) => k),
};
report.finishedAt = new Date().toISOString();
await fs.mkdir('.tmf-run/artifacts', { recursive: true });
await fs.writeFile('.tmf-run/artifacts/hf-peft-e2e.json', JSON.stringify(report, null, 2));
console.log(JSON.stringify(report.summary, null, 1));
process.exit(report.summary.failedSteps.length === 0 ? 0 : 1);
