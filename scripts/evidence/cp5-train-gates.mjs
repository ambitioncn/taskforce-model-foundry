// R3 acceptance run: step-budgeted (deterministic, not wall-clock) training ->
// independent baseline/tuned evaluation -> DEFAULT_GATES verdict -> reproducible report.
import { writeSyntheticCorpus } from '../../src/data/synth.mjs';
import { runDataPipeline } from '../../src/data/pipeline.mjs';
import { buildTokenizer, encodeRecords, countTokens } from '../../src/train/dataset.mjs';
import builtin from '../../src/train/backends/builtin-js.mjs';
import { writeJson } from '../../src/core/fsx.mjs';
import { relativeImprovement } from '../../src/eval/metrics.mjs';
import { evaluateGates, DEFAULT_GATES } from '../../src/eval/gates.mjs';

const PRETRAIN_STEPS = Number(process.env.TMF_PRETRAIN_STEPS || 150);
const LORA_STEPS = Number(process.env.TMF_LORA_STEPS || 700);
const HARD_SECONDS = Number(process.env.TMF_HARD_SECONDS || 3600); // safety valve only
const outFile = process.env.TMF_EVIDENCE_OUT || '.tmf-run/cp5-train-gates-report.json';
const MODEL_CONFIG = { dModel: 64, nLayers: 2, nHeads: 4, dFF: 128, maxSeqLen: 128 };

const corpus = await writeSyntheticCorpus('.tmf-run/corpus');
const pre = await runDataPipeline({
  datasets: corpus.pretrainDatasets,
  config: { split: { train: 0.9, validation: 0.1, holdout: 0, requireNonEmptyHoldout: false } },
});
const task = await runDataPipeline({ datasets: corpus.datasets });

const tokenizer = buildTokenizer([...pre.accepted, ...task.accepted], { maxVocab: 128 });
const preEx = encodeRecords(pre.splits.train, tokenizer, { maxSeqLen: 128 }).examples;
const trainEx = encodeRecords(task.splits.train, tokenizer, { maxSeqLen: 128 }).examples;
const valEx = encodeRecords(task.splits.validation, tokenizer, { maxSeqLen: 128 }).examples;
const holdEx = encodeRecords(task.splits.holdout, tokenizer, { maxSeqLen: 128 }).examples;

async function exactMatch(handle, records) {
  let hit = 0;
  const samples = [];
  for (const r of records) {
    const got = await builtin.generate({ handle, prompt: r.prompt, maxNewTokens: 34 });
    if (got === r.completion) hit += 1;
    if (samples.length < 3) samples.push({ prompt: r.prompt, got, want: r.completion });
  }
  return { accuracy: Number((hit / records.length).toFixed(6)), hit, total: records.length, samples };
}

const handle = await builtin.createModel({ modelConfig: MODEL_CONFIG, tokenizer, seed: 7 });
const baseParamCount = handle.model.parameterCount();

const pretrain = await builtin.train({
  handle, examples: preEx, mode: 'pretrain',
  hyper: { lr: 0.006, batchSize: 8 },
  budget: { maxSteps: PRETRAIN_STEPS, maxSeconds: HARD_SECONDS },
  seed: 'pre',
});
const baseFile = '.tmf-run/ckpt5/base.json';
await builtin.saveModel({ handle, file: baseFile, kind: 'base', meta: { stage: 'pretrain' } });
const baseHoldout = await builtin.evaluate({ handle, examples: holdEx });
const baseVal = await builtin.evaluate({ handle, examples: valEx });
const baseExact = await exactMatch(handle, task.splits.holdout);

const lora = await builtin.train({
  handle, examples: trainEx, mode: 'lora',
  hyper: { lr: 0.02, batchSize: 8, lora: { rank: 8, alpha: 16, targets: ['attn.q', 'attn.v', 'ffn.in', 'ffn.out'] } },
  budget: { maxSteps: LORA_STEPS, maxSeconds: HARD_SECONDS },
  seed: 'lora',
});
const adapterFile = '.tmf-run/ckpt5/adapter.json';
await builtin.saveModel({ handle, file: adapterFile, kind: 'adapter', meta: { stage: 'lora' } });
const tunedHoldout = await builtin.evaluate({ handle, examples: holdEx });
const tunedVal = await builtin.evaluate({ handle, examples: valEx });
const tunedExact = await exactMatch(handle, task.splits.holdout);

const reloaded = await builtin.loadModel({ file: baseFile, attachAdapter: adapterFile });
const reloadedHoldout = await builtin.evaluate({ handle: reloaded, examples: holdEx });
const reloadedExact = await exactMatch(reloaded, task.splits.holdout);

const metrics = {
  baseline: {
    holdout: { loss: baseHoldout.loss, perplexity: baseHoldout.perplexity, exactMatch: { accuracy: baseExact.accuracy, hit: baseExact.hit, total: baseExact.total } },
    validation: { loss: baseVal.loss, perplexity: baseVal.perplexity },
  },
  tuned: {
    holdout: { loss: tunedHoldout.loss, perplexity: tunedHoldout.perplexity, exactMatch: { accuracy: tunedExact.accuracy, hit: tunedExact.hit, total: tunedExact.total } },
    validation: { loss: tunedVal.loss, perplexity: tunedVal.perplexity },
  },
  improvement: {
    holdoutLossRelative: relativeImprovement(baseHoldout.loss, tunedHoldout.loss),
    validationLossRelative: relativeImprovement(baseVal.loss, tunedVal.loss),
    exactMatchAbsolute: Number((tunedExact.accuracy - baseExact.accuracy).toFixed(6)),
  },
};
const verdict = evaluateGates(metrics, DEFAULT_GATES);

const report = {
  format: 'tmf-cp5-train-gates-report-v1',
  generatedAt: new Date().toISOString(),
  reproduce: {
    command: 'node scripts/evidence/cp5-train-gates.mjs',
    env: { TMF_PRETRAIN_STEPS: PRETRAIN_STEPS, TMF_LORA_STEPS: LORA_STEPS },
    seeds: { model: 7, pretrain: 'pre', lora: 'lora' },
    budgetKind: 'steps',
  },
  data: {
    pretrainTrain: pre.splits.train.length,
    taskTrain: task.splits.train.length,
    taskValidation: task.splits.validation.length,
    taskHoldout: task.splits.holdout.length,
    pretrainTokens: countTokens(preEx),
    taskTokens: countTokens(trainEx),
    vocab: tokenizer.size,
    leakageOk: task.report.leakage.ok,
  },
  model: { baseParamCount, config: MODEL_CONFIG },
  pretrain: pretrain.metrics,
  lora: lora.metrics,
  metrics,
  reload: {
    holdout: reloadedHoldout,
    exactMatch: { accuracy: reloadedExact.accuracy, hit: reloadedExact.hit, total: reloadedExact.total },
    equivalent: Math.abs(reloadedHoldout.loss - tunedHoldout.loss) < 1e-9 && reloadedExact.hit === tunedExact.hit,
  },
  gates: verdict,
  samples: tunedExact.samples,
};
await writeJson(outFile, report);
console.log(JSON.stringify({
  pretrainStop: pretrain.metrics.stopReason, pretrainSteps: pretrain.metrics.steps,
  loraStop: lora.metrics.stopReason, loraSteps: lora.metrics.steps,
  gates: verdict.summary, failed: verdict.failed,
  tunedHoldoutLoss: tunedHoldout.loss, tunedExact: tunedExact.accuracy, baseExact: baseExact.accuracy,
  reloadEquivalent: report.reload.equivalent, out: outFile,
}, null, 2));
if (!verdict.passed) process.exitCode = 3;
