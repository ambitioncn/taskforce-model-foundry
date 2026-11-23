// Bounded, harness-independent experiment controller. The operator freezes a
// campaign before execution; this module never invents data, gates or model
// configurations and never exports/publishes a candidate on its own.
import path from 'node:path';
import { readJson, exists } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { loadConfig } from '../config.mjs';
import { train, evaluate } from '../api.mjs';
import { readMetric } from '../eval/gates.mjs';
import { assertReportIntegrity } from '../jobs/evaluate.mjs';
import { runJob, readJobState, readAuditLog } from '../jobs/runner.mjs';

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
  exactKeys(plan, ['format', 'id', 'candidates', 'budget', 'policy'], 'campaign');
  if (plan.format !== CAMPAIGN_FORMAT) throw new Error(`campaign.format must be ${CAMPAIGN_FORMAT}`);
  if (!ID.test(plan.id ?? '')) throw new Error('campaign.id must be a safe identifier');
  if (!Array.isArray(plan.candidates) || plan.candidates.length === 0) throw new Error('campaign.candidates must be a nonempty array');
  exactKeys(plan.budget, ['maxTrials', 'maxTotalTrainSteps', 'deadlineAt'], 'campaign.budget');
  const maxTrials = positiveInt(plan.budget.maxTrials, 'campaign.budget.maxTrials');
  const maxTotalTrainSteps = positiveInt(plan.budget.maxTotalTrainSteps, 'campaign.budget.maxTotalTrainSteps');
  if (maxTrials > plan.candidates.length) throw new Error('campaign.budget.maxTrials exceeds candidate count');
  const deadline = Date.parse(plan.budget.deadlineAt);
  if (!Number.isFinite(deadline) || !/Z$/.test(plan.budget.deadlineAt)) throw new Error('campaign.budget.deadlineAt must be a UTC ISO timestamp');
  exactKeys(plan.policy, ['stopOnTechnicalPass', 'rankMetric', 'direction'], 'campaign.policy');
  if (typeof plan.policy.stopOnTechnicalPass !== 'boolean') throw new Error('campaign.policy.stopOnTechnicalPass must be boolean');
  if (typeof plan.policy.rankMetric !== 'string' || !/^[a-zA-Z][a-zA-Z0-9.]*$/.test(plan.policy.rankMetric)) throw new Error('campaign.policy.rankMetric must be a metric path');
  if (!RANK_DIRECTIONS.has(plan.policy.direction)) throw new Error('campaign.policy.direction must be min or max');

  const seen = new Set();
  const snapshots = [];
  let comparableDigest = null;
  let runRoot = null;
  for (const [index, candidate] of plan.candidates.entries()) {
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
  if (snapshots[0].steps > maxTotalTrainSteps) throw new Error('campaign budget cannot fund the first candidate');
  return { file, baseDir, plan, snapshots, runRoot, input: { planSha256: await sha256File(file), snapshots, comparableDigest } };
}

function previousCandidateOutputs(outputs, snapshots, index) {
  return snapshots.slice(0, index).map((item) => outputs[`candidate-${item.id}`]).filter(Boolean);
}

/** Runs a frozen, bounded campaign; replay skips every completed trial. */
export async function runCampaign({ planFile, signal = null, resumeCancelled = false } = {}) {
  const loaded = await loadCampaign(planFile);
  const { plan, snapshots, runRoot, input } = loaded;
  const steps = snapshots.map((candidate, index) => ({
    name: `candidate-${candidate.id}`,
    run: async ({ outputs }) => {
      const previous = previousCandidateOutputs(outputs, snapshots, index);
      if (index >= plan.budget.maxTrials) return { id: candidate.id, state: 'skipped_trial_limit' };
      if (plan.policy.stopOnTechnicalPass && previous.some((item) => item.state === 'technical_passed')) {
        return { id: candidate.id, state: 'skipped_goal_reached' };
      }
      const spent = previous.reduce((sum, item) => sum + (item.stepsCharged ?? 0), 0);
      if (spent + candidate.steps > plan.budget.maxTotalTrainSteps) return { id: candidate.id, state: 'skipped_step_budget' };
      if (Date.now() >= Date.parse(plan.budget.deadlineAt)) return { id: candidate.id, state: 'skipped_deadline' };
      const trainResult = await train({ configFile: candidate.configFile, baseDir: loaded.baseDir,
        jobId: candidate.jobId, signal, resume: true, resumeCancelled });
      if (trainResult.job.state !== 'succeeded') throw new Error(`${candidate.id}: training did not succeed`);
      const reportFile = path.join(runRoot, candidate.jobId, 'eval-report.json');
      let report = await exists(reportFile) ? await readJson(reportFile) : null;
      if (report) {
        assertReportIntegrity(report, { jobId: candidate.jobId });
        if (report.lineage?.externalQualityContractDigest !== candidate.qualityDigest ||
            report.lineage?.gateDefinitionsDigest !== candidate.gatesDigest) {
          throw new Error(`${candidate.id}: existing report has incompatible frozen gates or quality lineage`);
        }
      } else {
        ({ report } = await evaluate({ configFile: candidate.configFile, baseDir: loaded.baseDir,
          jobId: candidate.jobId, reportFile }));
      }
      const score = readMetric(report.metrics, plan.policy.rankMetric);
      const passed = Boolean(report.verdict?.deliveryEligible);
      if (passed && (typeof score !== 'number' || !Number.isFinite(score))) {
        throw new Error(`${candidate.id}: rank metric ${plan.policy.rankMetric} missing or nonfinite`);
      }
      return { id: candidate.id, jobId: candidate.jobId,
        state: passed ? 'technical_passed' : 'technical_failed',
        externalQualityRequired: candidate.externalQualityRequired,
        reportFile, reportDigest: report.reportDigest,
        score: typeof score === 'number' && Number.isFinite(score) ? score : null,
        stepsCharged: candidate.steps };
    },
  }));
  steps.push({ name: 'decision', run: async ({ outputs }) => {
    const trials = snapshots.map((candidate) => outputs[`candidate-${candidate.id}`]);
    const passed = trials.filter((item) => item.state === 'technical_passed');
    const ordered = passed.sort((a, b) => plan.policy.direction === 'min' ? a.score - b.score : b.score - a.score);
    const selected = ordered[0] ?? null;
    return { status: selected ? (selected.externalQualityRequired ? 'awaiting_external_quality' : 'technical_passed') : 'no_candidate_passed',
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
