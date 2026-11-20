// Training job: data preparation, base checkpoint, and post-training run as
// one auditable job.
//
// The point of routing training through `runJob` is that the operator gets the
// same guarantees as every other foundry job -- durable state, a reviewable
// audit log, resume at step granularity, cooperative cancellation and hard
// budget caps -- without the training code having to re-implement any of it.
//
// Each step persists its real artifacts to disk and returns only small
// references (paths, digests, counts). That is deliberate: a resumed attempt
// reloads from those artifacts instead of trusting in-memory trainer state,
// so "resume" is the same code path as "load a checkpoint someone else wrote".

import path from 'node:path';
import { runJob, JobCancelledError } from './runner.mjs';
import { runDataPipeline } from '../data/pipeline.mjs';
import { buildTokenizer, encodeRecords } from '../train/dataset.mjs';
import { ensureDir, readJsonl, writeJson } from '../core/fsx.mjs';
import { sha256Json } from '../core/hash.mjs';

export const TRAIN_JOB_KIND = 'train';

export const DEFAULT_TRAIN_JOB_OPTIONS = {
  mode: 'lora',
  seed: 'tmf-train',
  maxVocab: 256,
  maxSeqLen: 96,
};

function splitFile(dataDir, name) {
  return path.join(dataDir, `${name}.jsonl`);
}

// Charging tokens as they are actually consumed is what makes `maxTokens` a
// real cap: the throw lands inside the training loop and the runner records
// state=budget_exceeded instead of discovering the overspend afterwards.
function makeStepHook({ signal, audit, budget, auditEveryStep }) {
  let lastCharged = 0;
  return async (entry) => {
    if (signal?.aborted) throw new JobCancelledError('cancelled during training step');
    const delta = Math.max(0, entry.tokens - lastCharged);
    lastCharged = entry.tokens;
    budget.charge({ tokens: delta });
    if (auditEveryStep > 0 && entry.step % auditEveryStep === 0) {
      await audit('train.step', { step: entry.step, loss: entry.loss, lr: entry.lr, tokens: entry.tokens });
    }
  };
}

/**
 * Run a full data -> base -> post-training job under the auditable runner.
 *
 * Returns the runner's job record; `result` holds the artifact references the
 * evaluation and export jobs consume (data report, base checkpoint, adapter
 * or tuned checkpoint, and the real training metrics).
 */
export async function runTrainingJob({
  jobId,
  root,
  backend,
  datasets,
  baseDir = process.cwd(),
  dataConfig = {},
  modelConfig = {},
  hyper = {},
  trainBudget = {},
  mode = DEFAULT_TRAIN_JOB_OPTIONS.mode,
  seed = DEFAULT_TRAIN_JOB_OPTIONS.seed,
  maxVocab = DEFAULT_TRAIN_JOB_OPTIONS.maxVocab,
  maxSeqLen = DEFAULT_TRAIN_JOB_OPTIONS.maxSeqLen,
  budget = {},
  signal = null,
  auditEveryStep = 1,
  resume = true,
  resumeCancelled = false,
  now = () => Date.now(),
} = {}) {
  if (!jobId) throw new Error('runTrainingJob requires jobId');
  if (!root) throw new Error('runTrainingJob requires root');
  if (!backend) throw new Error('runTrainingJob requires a backend');
  if (!Array.isArray(datasets) || datasets.length === 0) {
    throw new Error('runTrainingJob requires at least one dataset');
  }
  if (!backend.capabilities?.modes?.includes(mode)) {
    throw new Error(`backend "${backend.id}" does not support mode "${mode}"`);
  }

  // Artifacts live in the same directory the runner uses for this job's state
  // and audit log, so one directory is the whole durable record of the job.
  const jobDir = path.join(root, jobId);
  const dataDir = path.join(jobDir, 'data');
  const modelDir = path.join(jobDir, 'model');

  // The input digest binds this jobId to its configuration: the runner refuses
  // to resume a job whose inputs changed, which is what stops a "resume" from
  // silently becoming a different experiment.
  const input = {
    kind: TRAIN_JOB_KIND,
    backend: backend.id,
    datasets: datasets.map((d) => ({ name: d.name, path: d.path, format: d.format ?? null })),
    dataConfig,
    modelConfig,
    hyper,
    trainBudget,
    mode,
    seed,
    maxVocab,
    maxSeqLen,
  };

  const steps = [
    {
      name: 'data',
      run: async ({ audit }) => {
        await ensureDir(dataDir);
        const { report, splits, provenance } = await runDataPipeline({
          datasets,
          outDir: dataDir,
          baseDir,
          config: dataConfig,
        });
        const reportFile = path.join(dataDir, 'data-report.json');
        await writeJson(reportFile, { report, provenance });
        await audit('data.prepared', {
          splits: Object.fromEntries(Object.entries(splits).map(([k, v]) => [k, v.length])),
          accepted: report.accepted ?? null,
        });
        return {
          dataDir,
          reportFile,
          reportDigest: sha256Json(report),
          splits: Object.fromEntries(Object.entries(splits).map(([k, v]) => [k, v.length])),
        };
      },
    },
    {
      name: 'base',
      run: async ({ outputs, audit }) => {
        const trainRecords = await readJsonl(splitFile(outputs.data.dataDir, 'train'));
        if (trainRecords.length === 0) throw new Error('data step produced an empty train split');
        // Tokenizer ownership is a backend capability, not a job assumption: a
        // backend whose model carries its own tokenizer (hf-peft) must not be
        // handed a vocabulary built from this split, because ids from a JS
        // vocabulary mean nothing to that model's embedding table.
        const ownsTokenizer = backend.capabilities?.ownsTokenizer === true;
        const tokenizer = ownsTokenizer ? null : buildTokenizer(trainRecords, { maxVocab });
        const handle = await backend.createModel({
          modelConfig,
          tokenizer,
          seed: 1,
          modelId: modelConfig.modelId ?? modelConfig.baseModel ?? null,
          outDir: modelDir,
        });
        const file = path.join(modelDir, 'base.json');
        const saved = await backend.saveModel({
          handle,
          file,
          kind: 'base',
          meta: { jobId, dataReportDigest: outputs.data.reportDigest },
        });
        const fingerprint = backend.fingerprint(handle);
        const vocabSize = tokenizer ? tokenizer.size : null;
        await audit('base.saved', {
          file: saved.file,
          sha256: saved.sha256,
          vocabSize,
          tokenizerOwner: ownsTokenizer ? 'backend' : 'job',
          fingerprint,
        });
        return { file: saved.file, sha256: saved.sha256, fingerprint, vocabSize, tokenizerOwner: ownsTokenizer ? 'backend' : 'job' };
      },
    },
    {
      name: 'train',
      run: async ({ outputs, audit, budget: jobBudget }) => {
        const trainRecords = await readJsonl(splitFile(outputs.data.dataDir, 'train'));
        // Reloading the base from disk (rather than reusing the handle built in
        // the previous step) is what makes a resumed attempt equivalent to a
        // fresh process picking the job up.
        const handle = await backend.loadModel({ file: outputs.base.file });
        // One call for both backends: prepareDataset() is the single place that
        // knows whether ids come from a job-built vocabulary or from the
        // model's own tokenizer. The fallback keeps older backends working.
        const prepared = typeof backend.prepareDataset === 'function'
          ? await backend.prepareDataset({ handle, records: trainRecords, maxSeqLen, outDir: modelDir })
          : { ...encodeRecords(trainRecords, handle.tokenizer, { maxSeqLen }), tokens: null, tokenizer: 'job-built' };
        const examples = prepared.examples ?? [];
        const skipped = prepared.skipped ?? [];
        if (examples.length === 0) throw new Error('no train examples survived encoding');
        const beforeFingerprint = backend.fingerprint(handle);
        await audit('train.started', {
          mode,
          examples: examples.length,
          skipped: skipped.length,
          tokens: prepared.tokens ?? null,
          tokenizer: prepared.tokenizer ?? 'job-built',
          baseFingerprint: beforeFingerprint,
        });

        const trained = await backend.train({
          handle,
          examples,
          examplesFile: prepared.examplesFile ?? null,
          mode,
          hyper,
          budget: trainBudget,
          seed,
          outDir: modelDir,
          signal,
          resume: null,
          hooks: { onStep: makeStepHook({ signal, audit, budget: jobBudget, auditEveryStep }) },
        });
        const { metrics, history, stopReason } = trained;
        // A backend may return a moved handle (hf-peft points it at the adapter
        // directory the worker just wrote); saving the pre-train handle would
        // persist a checkpoint that does not reference the trained weights.
        const trainedHandle = trained.handle ?? handle;

        const historyFile = path.join(modelDir, 'train-history.json');
        await writeJson(historyFile, { jobId, mode, metrics, history });
        const isAdapter = mode === 'lora';
        const file = path.join(modelDir, isAdapter ? 'adapter.json' : 'tuned.json');
        const saved = await backend.saveModel({
          handle: trainedHandle,
          file,
          kind: isAdapter ? 'adapter' : 'base',
          meta: {
            jobId,
            mode,
            baseCheckpoint: outputs.base.file,
            baseFingerprint: beforeFingerprint,
            metrics,
          },
        });
        await audit('train.finished', {
          file: saved.file,
          sha256: saved.sha256,
          stopReason,
          steps: metrics.steps,
          finalLoss: metrics.finalLoss,
          parameterL2Delta: metrics.parameterL2Delta,
        });
        if (!(metrics.parameterL2Delta > 0)) {
          throw new Error('training reported zero parameter displacement; weights did not change');
        }
        return {
          checkpoint: saved.file,
          sha256: saved.sha256,
          kind: saved.kind,
          historyFile,
          baseFingerprint: beforeFingerprint,
          metrics,
          stopReason,
        };
      },
    },
    {
      // The runner's `result` is the last step's output, so the job ends with a
      // step whose only work is to collect the artifact references and write
      // them down as a manifest a downstream job can consume directly.
      name: 'finalize',
      run: async ({ outputs, audit }) => {
        const manifest = {
          jobId,
          kind: TRAIN_JOB_KIND,
          backend: backend.id,
          mode,
          data: outputs.data,
          base: outputs.base,
          train: outputs.train,
        };
        const manifestFile = path.join(jobDir, 'train-manifest.json');
        await writeJson(manifestFile, manifest);
        await audit('train.manifest', { manifestFile, digest: sha256Json(manifest) });
        return { ...manifest, manifestFile };
      },
    },
  ];

  return runJob({ jobId, kind: TRAIN_JOB_KIND, root, steps, input, budget, signal, resume, resumeCancelled, now });
}

/**
 * Artifact references a downstream evaluation or export job needs. Throws when
 * the job did not actually finish, so a caller cannot mistake a cancelled or
 * over-budget job for a trained model.
 */
export function trainingArtifacts(job) {
  if (job?.state !== 'succeeded') {
    throw new Error(`training job ${job?.jobId ?? '<unknown>'} is ${job?.state ?? 'missing'}, not succeeded`);
  }
  const { data, base, train } = job.result ?? {};
  if (!data || !base || !train) throw new Error('training job result is missing step outputs');
  return {
    jobId: job.jobId,
    dataDir: data.dataDir,
    dataReportFile: data.reportFile,
    baseCheckpoint: base.file,
    adapterCheckpoint: train.kind === 'adapter' ? train.checkpoint : null,
    tunedCheckpoint: train.kind === 'adapter' ? null : train.checkpoint,
    metrics: train.metrics,
  };
}
