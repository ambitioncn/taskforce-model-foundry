// The programmatic API.
//
// Everything the CLI does, it does through this module, so an embedder (the
// OpenClaw / Hermes adapters included) gets exactly the same behaviour without
// shelling out. Functions here return plain JSON-serializable records.

import path from 'node:path';
import process from 'node:process';
import { loadConfig, validateConfig, exampleConfig, ConfigError } from './config.mjs';
import { loadDefaultBackends, probeBackends, resolveBackend, listBackends } from './train/backends/index.mjs';
import { runTrainingJob, trainingArtifacts } from './jobs/train.mjs';
import { readJobState, readAuditLog } from './jobs/runner.mjs';
import { writeSyntheticCorpus } from './data/synth.mjs';
import { runEvaluationJob, assertReportIntegrity } from './jobs/evaluate.mjs';
import { DEFAULT_GATES } from './eval/gates.mjs';
import { encodeRecords } from './train/dataset.mjs';
import { exportModel as packageExport } from './export/package.mjs';
import { exists, readJson, readJsonl } from './core/fsx.mjs';
import { VERSION } from './version.mjs';
import { sha256Json, sha256File } from './core/hash.mjs';
import { verifyExternalQuality } from './eval/external-quality.mjs';
import { prepareScreenplayQuality as prepareScreenplayInputs, finalizeScreenplayQuality as finalizeScreenplayInputs } from './eval/screenplay-quality.mjs';

export const PRODUCT = 'taskforce-model-foundry';
const MIN_NODE_MAJOR = 20;

let backendsLoaded = false;
async function ensureBackends() {
  if (!backendsLoaded) {
    await loadDefaultBackends();
    backendsLoaded = true;
  }
  return listBackends();
}

/**
 * Environment and configuration self-check.
 *
 * Reports every check rather than throwing on the first failure: `ok` is false
 * when any required check fails, while optional checks (an absent GPU backend,
 * a missing config file when none was requested) are reported as warnings.
 */
export async function doctor({ baseDir = process.cwd(), configFile = null, requireConfig = false } = {}) {
  const checks = [];
  const nodeMajor = Number(process.versions.node.split('.')[0]);
  checks.push({
    id: 'node-version',
    level: 'required',
    ok: nodeMajor >= MIN_NODE_MAJOR,
    detail: `node ${process.versions.node} (requires >=${MIN_NODE_MAJOR})`,
  });

  await ensureBackends();
  const probes = await probeBackends();
  const availableBackends = probes.filter((p) => p.available).map((p) => p.id);
  checks.push({
    id: 'trainer-backend',
    level: 'required',
    ok: availableBackends.length > 0,
    detail: availableBackends.length > 0
      ? `available: ${availableBackends.join(', ')}`
      : 'no trainer backend is available in this environment',
  });
  for (const probe of probes.filter((p) => !p.available)) {
    checks.push({
      id: `backend:${probe.id}`,
      level: 'optional',
      ok: false,
      detail: `unavailable: ${probe.reason ?? 'unknown reason'}`,
    });
  }

  let config = null;
  let configPath = null;
  let configErrors = [];
  try {
    const loaded = await loadConfig({ file: configFile, baseDir });
    config = loaded.config;
    configPath = loaded.file;
    checks.push({ id: 'config', level: 'required', ok: true, detail: `valid: ${path.relative(baseDir, loaded.file) || loaded.file}` });
  } catch (err) {
    if (!(err instanceof ConfigError)) throw err;
    configErrors = err.errors;
    checks.push({
      id: 'config',
      level: requireConfig || configFile ? 'required' : 'optional',
      ok: false,
      detail: err.message,
    });
  }

  if (config) {
    for (const dataset of config.data.datasets) {
      const file = path.resolve(config.baseDir, dataset.path);
      checks.push({
        id: `dataset:${dataset.name}`,
        level: 'required',
        ok: await exists(file),
        detail: (await exists(file)) ? file : `missing dataset file: ${file}`,
      });
    }
    checks.push({
      id: 'backend-supports-mode',
      level: 'required',
      ok: probes.some((p) => p.id === config.backend && p.available && p.capabilities?.modes?.includes(config.mode)),
      detail: `backend "${config.backend}" must be available and support mode "${config.mode}"`,
    });
  }

  const failed = checks.filter((c) => c.level === 'required' && !c.ok);
  return {
    product: PRODUCT,
    ok: failed.length === 0,
    node: process.versions.node,
    platform: `${process.platform}-${process.arch}`,
    backends: probes.map(({ id, available, reason = null, capabilities }) => ({ id, available, reason, capabilities })),
    configFile: configPath,
    configErrors,
    checks,
    failed: failed.map((c) => c.id),
  };
}

/** Generate a synthetic corpus so a first run needs no private data. */
export async function synthesize({ baseDir = process.cwd(), outDir = 'data', ...options } = {}) {
  const dir = path.resolve(baseDir, outDir);
  const written = await writeSyntheticCorpus(dir, options);
  return { outDir: dir, ...written };
}

/** Run a real post-training job from a validated config. */
export async function train({
  config,
  configFile = null,
  baseDir = process.cwd(),
  jobId = null,
  signal = null,
  resume = true,
  resumeCancelled = false,
  auditEveryStep = 10,
} = {}) {
  const resolved = config ?? (await loadConfig({ file: configFile, baseDir })).config;
  await ensureBackends();
  const backend = await resolveBackend(resolved.backend);
  const job = await runTrainingJob({
    jobId: jobId ?? `train-${resolved.project}`,
    root: resolved.runRootAbsolute,
    backend,
    datasets: resolved.data.datasets,
    baseDir: resolved.baseDir,
    dataConfig: resolved.data.config,
    modelConfig: resolved.model,
    hyper: resolved.hyper,
    trainBudget: resolved.trainBudget,
    externalQuality: resolved.eval?.externalQuality ?? null,
    mode: resolved.mode,
    seed: resolved.seed,
    maxVocab: resolved.model.maxVocab,
    maxSeqLen: resolved.model.maxSeqLen,
    budget: resolved.budget,
    signal,
    resume,
    resumeCancelled,
    auditEveryStep,
  });
  return {
    job,
    artifacts: job.state === 'succeeded' ? trainingArtifacts(job) : null,
  };
}

/** Durable status of a job: its state record plus a tail of the audit log. */
export async function status({ config = null, configFile = null, baseDir = process.cwd(), jobId, auditTail = 10 } = {}) {
  if (!jobId) throw new Error('status requires jobId');
  const resolved = config ?? (await loadConfig({ file: configFile, baseDir })).config;
  const state = await readJobState(resolved.runRootAbsolute, jobId);
  if (!state) return { jobId, found: false, state: null, audit: [] };
  const audit = await readAuditLog(resolved.runRootAbsolute, jobId);
  return {
    jobId,
    found: true,
    state,
    auditCount: audit.length,
    audit: auditTail > 0 ? audit.slice(-auditTail) : [],
  };
}

const EVAL_SPLITS = ['holdout', 'validation', 'train'];

/**
 * Re-read the splits a training job wrote and encode them with the *saved*
 * tokenizer, so an evaluation run depends on checkpoints on disk rather than
 * on any state left behind by the trainer.
 */
async function loadEvalSplits({ backend, artifacts, maxSeqLen }) {
  const handle = await backend.loadModel({ file: artifacts.baseCheckpoint });
  const records = {};
  const encoded = {};
  for (const split of EVAL_SPLITS) {
    const file = path.join(artifacts.dataDir, `${split}.jsonl`);
    if (!(await exists(file))) continue;
    records[split] = await readJsonl(file);
    // Same backend contract the training job uses: whichever side owns the
    // tokenizer is the side that encodes, so evaluation never feeds a
    // job-built vocabulary to a model that carries its own.
    encoded[split] = typeof backend.prepareDataset === 'function'
      ? (await backend.prepareDataset({ handle, records: records[split], maxSeqLen })).examples
      : encodeRecords(records[split], handle.tokenizer, { maxSeqLen }).examples;
  }
  if (!encoded.holdout?.length) {
    throw new Error(`evaluate: no encodable holdout split under ${artifacts.dataDir}`);
  }
  return { records, encoded };
}

/** Resolve the training artifacts an evaluate/export call should work from. */
async function resolveArtifacts({ artifacts, config, jobId }) {
  if (artifacts) return artifacts;
  const id = jobId ?? `train-${config.project}`;
  const state = await readJobState(config.runRootAbsolute, id);
  if (!state) throw new Error(`no durable job state for "${id}" under ${config.runRootAbsolute}; run tmf train first`);
  // The state file keeps each step's output under completedSteps; rebuilding
  // the step map from it is what lets evaluate/export run in a fresh process
  // instead of only inside the one that trained.
  const result = Object.fromEntries((state.completedSteps ?? [])
    .filter((step) => step && typeof step === 'object' && step.name)
    .map((step) => [step.name, step.output]));
  return trainingArtifacts({ ...state, result });
}

/**
 * Independent evaluation of a finished training job: baseline vs tuned on a
 * holdout split that is checked for leakage, scored against the configured
 * gates, and written to a durable report.
 */
export async function evaluate({
  config,
  configFile = null,
  baseDir = process.cwd(),
  jobId = null,
  artifacts = null,
  gates = null,
  reportFile = null,
  maxNewTokens = 48,
} = {}) {
  const resolved = config ?? (await loadConfig({ file: configFile, baseDir })).config;
  await ensureBackends();
  const backend = await resolveBackend(resolved.backend);
  const resolvedArtifacts = await resolveArtifacts({ artifacts, config: resolved, jobId });
  const { records, encoded } = await loadEvalSplits({
    backend,
    artifacts: resolvedArtifacts,
    maxSeqLen: resolved.model.maxSeqLen,
  });
  const file = path.resolve(
    resolved.baseDir,
    reportFile ?? path.join(resolved.runRootAbsolute, resolvedArtifacts.jobId, 'eval-report.json'),
  );
  const report = await runEvaluationJob({
    backend,
    baseCheckpoint: resolvedArtifacts.baseCheckpoint,
    adapterCheckpoint: resolvedArtifacts.adapterCheckpoint,
    tunedCheckpoint: resolvedArtifacts.tunedCheckpoint ?? null,
    encoded,
    records,
    gates: gates ?? resolved.eval?.gates ?? DEFAULT_GATES,
    lineage: {
      jobId: resolvedArtifacts.jobId,
      project: resolved.project,
      mode: resolved.mode,
      seed: resolved.seed,
      dataDir: resolvedArtifacts.dataDir,
      dataReportFile: resolvedArtifacts.dataReportFile ?? null,
      externalQualityContractDigest: resolvedArtifacts.externalQuality ? sha256Json(resolvedArtifacts.externalQuality) : null,
    },
    reportFile: file,
    maxNewTokens,
    jobId: resolvedArtifacts.jobId,
  });
  return { artifacts: resolvedArtifacts, report };
}

async function readPinnedReport(file, baseDir) {
  const abs = path.resolve(baseDir ?? process.cwd(), file);
  if (!(await exists(abs))) throw new Error(`eval report file not found: ${abs}`);
  let parsed;
  try {
    parsed = await readJson(abs);
  } catch (err) {
    throw new Error(`eval report file is not valid JSON: ${abs} (${err.message})`);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`eval report file does not contain a report object: ${abs}`);
  }
  // Keep the on-disk location in the record so the bundle copies the reviewed
  // file itself instead of re-serializing it.
  return parsed.reportFile ? parsed : { ...parsed, reportFile: abs };
}

export async function screenplayQualityPrepare({
  configFile = null, baseDir = process.cwd(), jobId, evalReport,
  suiteFile, scriptsDir, baseOutputs, daptOutputs, candidateOutputs,
  cleanDaptReport, cleanDaptModel, outDir,
} = {}) {
  const { config } = await loadConfig({ file: configFile, baseDir });
  const artifacts = await resolveArtifacts({ config, jobId });
  const technical = typeof evalReport === 'string' ? await readPinnedReport(evalReport, config.baseDir) : evalReport;
  assertReportIntegrity(technical, { jobId });
  if (technical.verdict?.status !== 'passed' || !technical.verdict.deliveryEligible) {
    throw new Error('screenplay quality requires a passed technical evaluation; failed candidates must not consume the 120-question suite');
  }
  const contract = config.eval?.externalQuality;
  if (!contract || !artifacts.externalQuality || sha256Json(contract) !== sha256Json(artifacts.externalQuality) ||
    technical.lineage?.externalQualityContractDigest !== sha256Json(contract)) {
    throw new Error('screenplay quality contract was not frozen into training and technical evaluation');
  }
  const checkpoint = artifacts.adapterCheckpoint ?? artifacts.tunedCheckpoint;
  return prepareScreenplayInputs({
    contract, jobId, evalReport: technical, checkpointSha256: await sha256File(checkpoint),
    suiteFile, scriptsDir, baseOutputs, daptOutputs, candidateOutputs,
    cleanDaptReport, cleanDaptModel, outDir,
  });
}

export async function screenplayQualityFinalize({
  configFile = null, baseDir = process.cwd(), stageFile, judgeVerdicts,
  cleanCandidateReport, cleanCandidateModel, reportFile,
} = {}) {
  const { config } = await loadConfig({ file: configFile, baseDir });
  return finalizeScreenplayInputs({
    stageFile, contract: config.eval?.externalQuality, judgeVerdicts,
    cleanCandidateReport, cleanCandidateModel, reportFile,
  });
}

/**
 * Package a deliverable bundle. The gate lives inside exportModel, so a
 * failed or baseline-only evaluation cannot produce an export.
 */
export async function exportModel({
  config,
  configFile = null,
  baseDir = process.cwd(),
  jobId = null,
  artifacts = null,
  evalReport = null,
  qualityReport = null,
  outDir = null,
  version = null,
  evaluateIfMissing = true,
} = {}) {
  const resolved = config ?? (await loadConfig({ file: configFile, baseDir })).config;
  let resolvedArtifacts = artifacts;
  // Hosts (Hermes / OpenClaw tool calls) can only hand over JSON scalars, so
  // evalReport arrives as a path there while in-process callers pass the loaded
  // report object. Resolve the path form here, once, rather than in each host
  // adapter, so both entry points pin reports the same way.
  let report = typeof evalReport === 'string' ? await readPinnedReport(evalReport, resolved.baseDir ?? baseDir) : evalReport;
  if (!report) {
    if (!evaluateIfMissing) throw new Error('exportModel requires an evalReport (or evaluateIfMissing)');
    const run = await evaluate({ config: resolved, baseDir, jobId, artifacts });
    resolvedArtifacts = run.artifacts;
    report = run.report;
  } else {
    // An externally supplied report pins the bundle to numbers a human already
    // reviewed, so it has to prove it has not been edited since.
    assertReportIntegrity(report, { jobId });
    resolvedArtifacts = await resolveArtifacts({ artifacts, config: resolved, jobId });
  }
  let externalQuality = null;
  if (resolved.eval?.externalQuality) {
    if (!resolvedArtifacts.externalQuality || sha256Json(resolvedArtifacts.externalQuality) !== sha256Json(resolved.eval.externalQuality)) {
      throw new Error('external quality contract was not frozen into this training job; start a new job with the contract predeclared');
    }
    if (report.lineage?.externalQualityContractDigest !== sha256Json(resolved.eval.externalQuality)) {
      throw new Error('technical evaluation is not bound to the predeclared external quality contract');
    }
    externalQuality = await verifyExternalQuality({
      contract: resolved.eval.externalQuality,
      evidence: qualityReport,
      artifacts: resolvedArtifacts,
      evalReport: report,
      baseDir: resolved.baseDir,
    });
  } else if (qualityReport) {
    throw new Error('qualityReport supplied but eval.externalQuality was not predeclared in the training config');
  }
  const target = path.resolve(resolved.baseDir, outDir ?? path.join(resolved.runRootAbsolute, resolvedArtifacts.jobId, 'export'));
  const bundle = await packageExport({
    outDir: target,
    artifacts: resolvedArtifacts,
    evalReport: report,
    externalQuality,
    config: resolved,
    name: resolved.project,
    version: version ?? VERSION,
    toolkitVersion: VERSION,
  });
  return { artifacts: resolvedArtifacts, report, bundle };
}

export { loadConfig, validateConfig, exampleConfig, ConfigError, probeBackends, listBackends };
