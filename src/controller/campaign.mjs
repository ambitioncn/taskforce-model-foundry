// Bounded, harness-independent experiment controller. The operator freezes a
// campaign before execution. A proposal mode may generate hyperparameter
// candidates, but never invents data/gates or exports/publishes a model.
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { readJson, writeJson, ensureDir, exists } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { loadConfig } from '../config.mjs';
import { train, evaluate, qualityContext } from '../api.mjs';
import { verifyExternalQuality } from '../eval/external-quality.mjs';
import { readMetric } from '../eval/gates.mjs';
import { assertReportIntegrity } from '../jobs/evaluate.mjs';
import { runJob, readJobState, readAuditLog } from '../jobs/runner.mjs';
import { buildProposalPool, chooseNextProposal, chooseReflectiveProposal, reflectOnFailedTrial, gateFailureSeverity } from './proposals.mjs';

export const CAMPAIGN_FORMAT = 'tmf-campaign-v1';
const ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/;
const RANK_DIRECTIONS = new Set(['min', 'max']);

function exactKeys(value, allowed, where) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${where} must be an object`);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`${where}.${key} is not supported`);
}

function positiveInt(value, where) {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${where} must be a positive integer`);
  return value;
}

function confinedFile(baseDir, relative, where) {
  if (typeof relative !== 'string' || !relative || path.isAbsolute(relative)) throw new Error(`${where} must be a relative file path`);
  const full = path.resolve(baseDir, relative);
  if (full === baseDir || path.relative(baseDir, full).startsWith('..')) throw new Error(`${where} must remain under the campaign directory`);
  return full;
}

/** Validate and fingerprint all inputs before the first training step. */
export async function loadCampaign(planFile) {
  const file = path.resolve(planFile);
  const baseDir = path.dirname(file);
  const plan = await readJson(file);
  exactKeys(plan, ['format', 'id', 'candidates', 'proposal', 'budget', 'policy', 'qualityRunner'], 'campaign');
  if (plan.format !== CAMPAIGN_FORMAT) throw new Error(`campaign.format must be ${CAMPAIGN_FORMAT}`);
  if (!ID.test(plan.id ?? '')) throw new Error('campaign.id must be a safe identifier');
  const hasCandidates = plan.candidates !== undefined;
  if (hasCandidates === (plan.proposal !== undefined)) throw new Error('campaign requires exactly one of candidates or proposal');
  if (hasCandidates && (!Array.isArray(plan.candidates) || plan.candidates.length === 0)) throw new Error('campaign.candidates must be a nonempty array');
  exactKeys(plan.budget, ['maxTrials', 'maxTotalTrainSteps', 'deadlineAt'], 'campaign.budget');
  const maxTrials = positiveInt(plan.budget.maxTrials, 'campaign.budget.maxTrials');
  const maxTotalTrainSteps = positiveInt(plan.budget.maxTotalTrainSteps, 'campaign.budget.maxTotalTrainSteps');
  if (hasCandidates && maxTrials > plan.candidates.length) throw new Error('campaign.budget.maxTrials exceeds candidate count');
  const deadline = Date.parse(plan.budget.deadlineAt);
  if (!Number.isFinite(deadline) || !/Z$/.test(plan.budget.deadlineAt)) throw new Error('campaign.budget.deadlineAt must be a UTC ISO timestamp');
  exactKeys(plan.policy, ['stopOnTechnicalPass', 'rankMetric', 'direction'], 'campaign.policy');
  if (typeof plan.policy.stopOnTechnicalPass !== 'boolean') throw new Error('campaign.policy.stopOnTechnicalPass must be boolean');
  if (typeof plan.policy.rankMetric !== 'string' || !/^[a-zA-Z][a-zA-Z0-9.]*$/.test(plan.policy.rankMetric)) throw new Error('campaign.policy.rankMetric must be a metric path');
  if (!RANK_DIRECTIONS.has(plan.policy.direction)) throw new Error('campaign.policy.direction must be min or max');
  let qualityRunner = null;
  if (plan.qualityRunner) {
    exactKeys(plan.qualityRunner, ['moduleFile'], 'campaign.qualityRunner');
    const moduleFile = confinedFile(baseDir, plan.qualityRunner.moduleFile, 'campaign.qualityRunner.moduleFile');
    if (!(await exists(moduleFile))) throw new Error('campaign.qualityRunner.moduleFile is missing');
    qualityRunner = { moduleFile, moduleSha256: await sha256File(moduleFile) };
  }

  const seen = new Set();
  const snapshots = [];
  let comparableDigest = null;
  let runRoot = null;
  for (const [index, candidate] of (plan.candidates ?? []).entries()) {
    exactKeys(candidate, ['id', 'configFile'], `campaign.candidates[${index}]`);
    if (!ID.test(candidate.id ?? '') || seen.has(candidate.id)) throw new Error(`candidate id must be safe and unique: ${candidate.id}`);
    seen.add(candidate.id);
    const configFile = confinedFile(baseDir, candidate.configFile, `campaign.candidates[${index}].configFile`);
    const { config } = await loadConfig({ file: configFile, baseDir });
    if (!Array.isArray(config.eval?.gates) || config.eval.gates.length === 0) throw new Error(`${candidate.id}: eval.gates must be predeclared, not defaults`);
    const steps = positiveInt(config.hyper?.steps, `${candidate.id}: hyper.steps`);
    const datasetDigests = [];
    for (const dataset of config.data.datasets) {
      const datasetFile = path.resolve(config.baseDir, dataset.path);
      if (!(await exists(datasetFile))) throw new Error(`${candidate.id}: missing dataset ${dataset.path}`);
      datasetDigests.push({ name: dataset.name, path: dataset.path, sha256: await sha256File(datasetFile) });
    }
    const comparable = sha256Json({ project: config.project, backend: config.backend, mode: config.mode,
      model: config.model, data: config.data, datasets: datasetDigests, gates: config.eval.gates,
      externalQuality: config.eval.externalQuality ?? null });
    if (comparableDigest && comparableDigest !== comparable) throw new Error(`${candidate.id}: candidates must share project, base model, datasets and frozen gates`);
    if (runRoot && runRoot !== config.runRootAbsolute) throw new Error(`${candidate.id}: candidates must share runRoot`);
    comparableDigest = comparable;
    runRoot = config.runRootAbsolute;
    snapshots.push({ id: candidate.id, configFile, configSha256: await sha256File(configFile),
      datasetDigests, steps, jobId: `${plan.id}--${candidate.id}`,
      gatesDigest: sha256Json(config.eval.gates),
      qualityDigest: config.eval.externalQuality ? sha256Json(config.eval.externalQuality) : null,
      externalQualityRequired: Boolean(config.eval.externalQuality) });
  }
  let proposalPool = null;
  let baseConfigSha256 = null;
  if (plan.proposal) {
    const baseConfigFile = confinedFile(baseDir, plan.proposal.baseConfigFile, 'campaign.proposal.baseConfigFile');
    const baseRaw = await readJson(baseConfigFile);
    const { config } = await loadConfig({ file: baseConfigFile, baseDir });
    if (!Array.isArray(config.eval?.gates) || config.eval.gates.length === 0) throw new Error('proposal base: eval.gates must be predeclared');
    positiveInt(config.hyper?.steps, 'proposal base: hyper.steps');
    const datasetDigests = [];
    for (const dataset of config.data.datasets) {
      const datasetFile = path.resolve(config.baseDir, dataset.path);
      if (!(await exists(datasetFile))) throw new Error(`proposal base: missing dataset ${dataset.path}`);
      datasetDigests.push({ name: dataset.name, path: dataset.path, sha256: await sha256File(datasetFile) });
    }
    proposalPool = buildProposalPool({ proposal: plan.proposal, baseRaw, baseDir });
    if (maxTrials > proposalPool.pool.length) throw new Error('campaign.budget.maxTrials exceeds proposal combinations');
    runRoot = config.runRootAbsolute;
    comparableDigest = sha256Json({ project: config.project, backend: config.backend, mode: config.mode,
      model: config.model, data: config.data, datasets: datasetDigests, gates: config.eval.gates,
      externalQuality: config.eval.externalQuality ?? null });
    baseConfigSha256 = await sha256File(baseConfigFile);
    for (const item of proposalPool.pool) {
      const generatedFile = path.join(runRoot, 'campaigns', plan.id, 'proposals', `${item.id}.json`);
      if (await exists(generatedFile) && sha256Json(await readJson(generatedFile)) !== item.configSha256) {
        throw new Error(`${item.id}: generated candidate file was changed after proposal`);
      }
      snapshots.push({ ...item, jobId: `${plan.id}--${item.id}`,
        datasetDigests, gatesDigest: sha256Json(config.eval.gates),
        qualityDigest: config.eval.externalQuality ? sha256Json(config.eval.externalQuality) : null,
        externalQualityRequired: Boolean(config.eval.externalQuality) });
    }
  }
  if (snapshots[0].steps > maxTotalTrainSteps) throw new Error('campaign budget cannot fund the first candidate');
  if (qualityRunner && snapshots.some((item) => !item.externalQualityRequired)) {
    throw new Error('qualityRunner requires a frozen eval.externalQuality contract for every candidate');
  }
  if (qualityRunner) {
    const config = (await loadConfig({ file: proposalPool ? confinedFile(baseDir, plan.proposal.baseConfigFile,
      'campaign.proposal.baseConfigFile') : snapshots[0].configFile, baseDir })).config;
    if (config.eval.externalQuality.scorerSha256 !== qualityRunner.moduleSha256) {
      throw new Error('qualityRunner module SHA-256 must match frozen eval.externalQuality.scorerSha256');
    }
  }
  const planSha256 = await sha256File(file);
  const input = proposalPool
    ? { planSha256, baseConfigSha256, snapshots: snapshots.map(({ raw, config, ...item }) => item), comparableDigest, qualityRunner }
    : { planSha256, snapshots, comparableDigest, qualityRunner };
  return { file, baseDir, plan, snapshots, proposalPool, runRoot, input, qualityRunner };
}

function previousCandidateOutputs(outputs, snapshots, index) {
  return snapshots.slice(0, index).map((item) => outputs[`candidate-${item.id}`]).filter(Boolean);
}

async function materializeProposal(loaded, candidate) {
  if (!candidate.raw) return candidate.configFile;
  const dir = path.join(loaded.runRoot, 'campaigns', loaded.plan.id, 'proposals');
  await ensureDir(dir);
  const file = path.join(dir, `${candidate.id}.json`);
  if (await exists(file)) {
    if (sha256Json(await readJson(file)) !== candidate.configSha256) {
      throw new Error(`${candidate.id}: generated candidate file was changed after proposal`);
    }
  } else await writeJson(file, candidate.raw);
  return file;
}

/** Runs a frozen, bounded campaign; replay skips every completed trial. */
export async function runCampaign({ planFile, signal = null, resumeCancelled = false } = {}) {
  const loaded = await loadCampaign(planFile);
  const { plan, snapshots, proposalPool, runRoot, input, qualityRunner } = loaded;

  async function runTrial(candidate, previous) {
    if (plan.policy.stopOnTechnicalPass && previous.some((item) => item.state === (qualityRunner ? 'quality_passed' : 'technical_passed'))) {
      return { state: 'skipped_goal_reached' };
    }
    const spent = previous.reduce((sum, item) => sum + (item.stepsCharged ?? 0), 0);
    if (spent + candidate.steps > plan.budget.maxTotalTrainSteps) return { state: 'skipped_step_budget' };
    if (Date.now() >= Date.parse(plan.budget.deadlineAt)) return { state: 'skipped_deadline' };
    const configFile = await materializeProposal(loaded, candidate);
    const trainResult = await train({ config: candidate.config ?? null, configFile, baseDir: loaded.baseDir,
      jobId: candidate.jobId, signal, resume: true, resumeCancelled });
    if (trainResult.job.state !== 'succeeded') throw new Error(`${candidate.id}: training did not succeed`);
    const reportFile = path.join(runRoot, candidate.jobId, 'eval-report.json');
    let report = await exists(reportFile) ? await readJson(reportFile) : null;
    let artifacts = trainResult.artifacts;
    if (report) {
      assertReportIntegrity(report, { jobId: candidate.jobId });
      if (report.lineage?.externalQualityContractDigest !== candidate.qualityDigest ||
          report.lineage?.gateDefinitionsDigest !== candidate.gatesDigest) {
        throw new Error(`${candidate.id}: existing report has incompatible frozen gates or quality lineage`);
      }
    } else {
      ({ report, artifacts } = await evaluate({ config: candidate.config ?? null, configFile, baseDir: loaded.baseDir,
        jobId: candidate.jobId, reportFile }));
    }
    const technicalPassed = Boolean(report.verdict?.deliveryEligible);
    let quality = null;
    if (technicalPassed && qualityRunner) {
      if (await sha256File(qualityRunner.moduleFile) !== qualityRunner.moduleSha256) {
        throw new Error('quality runner changed after campaign submission');
      }
      const context = await qualityContext({ config: candidate.config ?? null, configFile, baseDir: loaded.baseDir,
        jobId: candidate.jobId, evalReport: report });
      const qualityReportFile = path.join(runRoot, candidate.jobId, 'quality-report.json');
      if (!(await exists(qualityReportFile))) {
        const module = await import(`${pathToFileURL(qualityRunner.moduleFile).href}?sha256=${qualityRunner.moduleSha256}`);
        if (typeof module.runQuality !== 'function') throw new Error('quality runner must export runQuality(context)');
        await module.runQuality({ ...context, configFile, reportFile: qualityReportFile,
          runRoot, baseDir: loaded.baseDir, jobId: candidate.jobId, signal });
      }
      quality = await verifyExternalQuality({ contract: context.contract, evidence: qualityReportFile,
        artifacts, evalReport: report, baseDir: loaded.baseDir, requirePass: false });
    }
    const measured = quality?.report ?? report;
    const score = readMetric(report.metrics, plan.policy.rankMetric);
    const passed = technicalPassed && (!qualityRunner || quality.gates.passed);
    if (passed && (typeof score !== 'number' || !Number.isFinite(score))) {
      throw new Error(`${candidate.id}: rank metric ${plan.policy.rankMetric} missing or nonfinite`);
    }
    const scoreValue = typeof score === 'number' && Number.isFinite(score) ? score : null;
    const gateSeverity = gateFailureSeverity((quality && !quality.gates.passed ? quality.gates : report.gates)?.results);
    const reflection = !passed && plan.proposal?.strategy === 'reflective'
      ? reflectOnFailedTrial({ pool: snapshots, previous, current: { ...candidate, score: scoreValue, gateSeverity },
        report: { ...measured, gates: quality && !quality.gates.passed ? quality.gates : report.gates },
        direction: plan.policy.direction }) : null;
    return { id: candidate.id, jobId: candidate.jobId, configFile,
      state: passed ? (qualityRunner ? 'quality_passed' : 'technical_passed') : (technicalPassed ? 'quality_failed' : 'technical_failed'),
      externalQualityRequired: candidate.externalQualityRequired,
      proposal: candidate.parameters ? { strategy: plan.proposal.strategy, parameters: candidate.parameters,
        parentId: candidate.parentId } : null,
      reflection,
      reportFile, reportDigest: report.reportDigest,
      qualityReportFile: quality?.file ?? null, qualityReportDigest: quality?.report.reportDigest ?? null,
      score: scoreValue, gateSeverity,
      stepsCharged: candidate.steps };
  }

  const steps = proposalPool
    ? Array.from({ length: plan.budget.maxTrials }, (_, index) => ({
      name: `trial-${index + 1}`,
      run: async ({ outputs }) => {
        const previous = Array.from({ length: index }, (_, i) => outputs[`trial-${i + 1}`]).filter(Boolean);
        if (plan.policy.stopOnTechnicalPass && previous.some((item) => item.state === (qualityRunner ? 'quality_passed' : 'technical_passed'))) {
          return { state: 'skipped_goal_reached' };
        }
        const candidate = plan.proposal.strategy === 'reflective'
          ? chooseReflectiveProposal({ pool: snapshots, baselineIndices: proposalPool.baselineIndices, previous })
          : chooseNextProposal({ pool: snapshots, baselineIndices: proposalPool.baselineIndices,
            previous, direction: plan.policy.direction });
        if (!candidate) return { state: [...previous].reverse().find((item) => item?.id)?.reflection?.action.method === 'stop_unactionable'
          ? 'skipped_unactionable_evidence' : 'skipped_exhausted' };
        return runTrial(candidate, previous);
      },
    }))
    : snapshots.map((candidate, index) => ({
      name: `candidate-${candidate.id}`,
      run: async ({ outputs }) => {
        const previous = previousCandidateOutputs(outputs, snapshots, index);
        if (index >= plan.budget.maxTrials) return { id: candidate.id, state: 'skipped_trial_limit' };
        return runTrial(candidate, previous);
      },
    }));
  steps.push({ name: 'decision', run: async ({ outputs }) => {
    const trials = proposalPool
      ? Array.from({ length: plan.budget.maxTrials }, (_, i) => outputs[`trial-${i + 1}`])
      : snapshots.map((candidate) => outputs[`candidate-${candidate.id}`]);
    const passed = trials.filter((item) => item.state === (qualityRunner ? 'quality_passed' : 'technical_passed'));
    const ordered = passed.sort((a, b) => plan.policy.direction === 'min' ? a.score - b.score : b.score - a.score);
    const selected = ordered[0] ?? null;
    return { status: selected ? (qualityRunner ? 'quality_passed' : selected.externalQualityRequired ? 'awaiting_external_quality' : 'technical_passed') : 'no_candidate_passed',
      selected, trials, stepsCharged: trials.reduce((sum, item) => sum + (item.stepsCharged ?? 0), 0),
      technicalOnly: true, exported: false };
  }});
  return runJob({ jobId: plan.id, kind: 'campaign', root: path.join(runRoot, 'campaigns'), steps,
    input, signal, resume: true, resumeCancelled });
}

export async function campaignStatus({ planFile } = {}) {
  const loaded = await loadCampaign(planFile);
  const root = path.join(loaded.runRoot, 'campaigns');
  const state = await readJobState(root, loaded.plan.id);
  return { found: Boolean(state), state, audit: state ? await readAuditLog(root, loaded.plan.id) : [] };
}
