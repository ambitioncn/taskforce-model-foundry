// Real training evidence run: synthetic corpus -> data pipeline -> pretrain -> LoRA post-train
// -> independent holdout evaluation -> checkpoint save/reload equivalence.
import { writeSyntheticCorpus } from '../../src/data/synth.mjs';
import { runDataPipeline } from '../../src/data/pipeline.mjs';
import { buildTokenizer, encodeRecords, countTokens } from '../../src/train/dataset.mjs';
import builtin from '../../src/train/backends/builtin-js.mjs';
import { writeJson } from '../../src/core/fsx.mjs';

const PRETRAIN_SECONDS = Number(process.env.TMF_PRETRAIN_SECONDS || 45);
const LORA_SECONDS = Number(process.env.TMF_LORA_SECONDS || 75);
const outFile = process.env.TMF_EVIDENCE_OUT || '.tmf-run/cp3-train-report.json';

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
  return { accuracy: hit / records.length, hit, total: records.length, samples };
}

const handle = await builtin.createModel({
  modelConfig: { dModel: 64, nLayers: 2, nHeads: 4, dFF: 128, maxSeqLen: 128 },
  tokenizer,
  seed: 7,
});
const baseParamCount = handle.model.parameterCount();

const pretrain = await builtin.train({
  handle, examples: preEx, mode: 'pretrain',
  hyper: { lr: 0.006, batchSize: 8 },
  budget: { maxSteps: 4000, maxSeconds: PRETRAIN_SECONDS },
  seed: 'pre',
});
const baseFile = '.tmf-run/ckpt/base.json';
await builtin.saveModel({ handle, file: baseFile, kind: 'base', meta: { stage: 'pretrain' } });
const baseHoldout = await builtin.evaluate({ handle, examples: holdEx });
const baseExact = await exactMatch(handle, task.splits.holdout);

const lora = await builtin.train({
  handle, examples: trainEx, mode: 'lora',
  hyper: { lr: 0.02, batchSize: 8, lora: { rank: 8, alpha: 16, targets: ['attn.q', 'attn.v', 'ffn.in', 'ffn.out'] } },
  budget: { maxSteps: 4000, maxSeconds: LORA_SECONDS },
  seed: 'lora',
});
const adapterFile = '.tmf-run/ckpt/adapter.json';
await builtin.saveModel({ handle, file: adapterFile, kind: 'adapter', meta: { stage: 'lora' } });
const tunedHoldout = await builtin.evaluate({ handle, examples: holdEx });
const tunedVal = await builtin.evaluate({ handle, examples: valEx });
const tunedExact = await exactMatch(handle, task.splits.holdout);

// reload equivalence: base checkpoint + adapter must reproduce the tuned metrics exactly
const reloaded = await builtin.loadModel({ file: baseFile, attachAdapter: adapterFile });
const reloadedHoldout = await builtin.evaluate({ handle: reloaded, examples: holdEx });
const reloadedExact = await exactMatch(reloaded, task.splits.holdout);

const report = {
  format: 'tmf-cp3-train-report-v1',
  generatedAt: new Date().toISOString(),
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
  model: { baseParamCount, config: { dModel: 64, nLayers: 2, nHeads: 4, dFF: 128, maxSeqLen: 128 } },
  pretrain: pretrain.metrics,
  lora: lora.metrics,
  evaluation: {
    baselineHoldout: baseHoldout,
    baselineExactMatch: { accuracy: baseExact.accuracy, hit: baseExact.hit, total: baseExact.total },
    tunedHoldout,
    tunedValidation: tunedVal,
    tunedExactMatch: { accuracy: tunedExact.accuracy, hit: tunedExact.hit, total: tunedExact.total },
    reloadedHoldout,
    reloadedExactMatch: { accuracy: reloadedExact.accuracy, hit: reloadedExact.hit, total: reloadedExact.total },
    reloadEquivalent:
      Math.abs(reloadedHoldout.loss - tunedHoldout.loss) < 1e-9 && reloadedExact.hit === tunedExact.hit,
    holdoutLossImprovement: Number((baseHoldout.loss - tunedHoldout.loss).toFixed(6)),
  },
  samples: tunedExact.samples,
  budgets: { pretrainSeconds: PRETRAIN_SECONDS, loraSeconds: LORA_SECONDS },
};
await writeJson(outFile, report);
console.log(JSON.stringify({
  pretrain: { firstLoss: pretrain.metrics.firstLoss, finalLoss: pretrain.metrics.finalLoss, steps: pretrain.metrics.steps, stop: pretrain.metrics.stopReason },
  lora: { firstLoss: lora.metrics.firstLoss, finalLoss: lora.metrics.finalLoss, steps: lora.metrics.steps, l2: lora.metrics.parameterL2Delta, stop: lora.metrics.stopReason },
  eval: report.evaluation,
  out: outFile,
}, null, 2));
