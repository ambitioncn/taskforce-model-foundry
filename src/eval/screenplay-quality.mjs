// CPU-only bridge to the existing G5 screenplay rule, promotion and blind-A/B
// scripts. It never generates model output or invents a judge verdict.
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { ensureDir, readJson, readJsonl, writeJson } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { evaluateGates } from './gates.mjs';
import { EXTERNAL_QUALITY_FORMAT, sealExternalQualityEvidence } from './external-quality.mjs';
import { SCREENPLAY_PROFILE, SCREENPLAY_GATES, screenplayGatesMatch } from './screenplay-profile.mjs';

const run = promisify(execFile);
export const DEFAULT_SCREENPLAY_SCRIPTS_DIR = fileURLToPath(new URL('../../scripts/screenplay-quality/', import.meta.url));
const SCRIPT_NAMES = {
  score: 'score_screenplay_generation_quality.py',
  table: 'build_screenplay_g5_promotion_table.py',
  blind: 'build_screenplay_g5_blind_ab_pairs.py',
};

function fail(message) { throw new Error(`screenplay quality: ${message}`); }
const finite = (value) => typeof value === 'number' && Number.isFinite(value);

async function python(script, args) {
  try {
    return (await run(process.env.TMF_PYTHON ?? 'python3', [script, ...args], { timeout: 120000, maxBuffer: 8 * 1024 * 1024 })).stdout;
  } catch (err) { fail(`${path.basename(script)} failed: ${(err.stderr || err.message).slice(-1000)}`); }
}

export async function screenplayScriptDigest(scriptsDir = DEFAULT_SCREENPLAY_SCRIPTS_DIR) {
  const hashes = {};
  for (const [role, name] of Object.entries(SCRIPT_NAMES)) hashes[role] = await sha256File(path.join(scriptsDir, name));
  return sha256Json(hashes);
}

export async function createScreenplayQualityContract({ suiteFile, baseOutputs, daptOutputs, cleanDaptReport,
  scriptsDir = DEFAULT_SCREENPLAY_SCRIPTS_DIR }) {
  const items = suiteItems(await readJson(suiteFile));
  await checkedOutputs(baseOutputs, items);
  await checkedOutputs(daptOutputs, items);
  const daptLoss = await readJson(cleanDaptReport);
  if (!Array.isArray(daptLoss?.results) || !daptLoss.results.some((row) => row.protocol === 'chat_wrapped' && finite(row.token_weighted_loss))) {
    fail('DAPT clean-validation report lacks a chat_wrapped loss');
  }
  if (!daptLoss.base_model || !daptLoss.val_jsonl || !daptLoss.tokenizer ||
      !daptLoss.model?.class || !daptLoss.model?.attn_implementation) {
    fail('DAPT clean-validation report lacks model/tokenizer/protocol provenance');
  }
  return {
    profile: SCREENPLAY_PROFILE,
    suiteSha256: await sha256File(suiteFile),
    scorerSha256: await screenplayScriptDigest(scriptsDir),
    references: {
      baseOutputs: await sha256File(baseOutputs),
      daptOutputs: await sha256File(daptOutputs),
      cleanDaptReport: await sha256File(cleanDaptReport),
    },
    gates: SCREENPLAY_GATES,
  };
}

async function source(file) { return { file: path.resolve(file), sha256: await sha256File(file) }; }
function suiteItems(suite) {
  const rows = Array.isArray(suite) ? suite : suite?.items;
  if (!Array.isArray(rows) || rows.length !== 120) fail('the frozen suite must contain exactly 120 items');
  const ids = rows.map((row) => row.id);
  if (new Set(ids).size !== 120 || rows.some((row) => typeof row.prompt !== 'string' || !row.prompt)) fail('suite IDs/prompts are invalid');
  return rows;
}

async function checkedOutputs(file, items) {
  const rows = await readJsonl(file);
  const byId = new Map(items.map((item) => [item.id, item]));
  if (rows.length !== items.length || new Set(rows.map((row) => row.id)).size !== items.length) fail(`${file}: incomplete or duplicate outputs`);
  for (const row of rows) {
    const item = byId.get(row.id);
    if (!item || row.prompt !== item.prompt || typeof row.output !== 'string') fail(`${file}: output does not match suite id/prompt ${row.id}`);
    if (!Number.isInteger(row.new_tokens) || row.new_tokens < 0 || row.max_new_tokens !== item.max_new_tokens ||
      typeof row.hit_max_new_tokens !== 'boolean' || row.hit_max_new_tokens !== (row.new_tokens >= row.max_new_tokens)) {
      fail(`${file}: generation budget/stop telemetry is invalid for ${row.id}`);
    }
  }
  return rows;
}

function requireProfile(contract) {
  if (contract?.profile !== SCREENPLAY_PROFILE || !screenplayGatesMatch(contract.gates)) fail('frozen screenplay-zh-v1 contract and all original thresholds are required');
  for (const key of ['baseOutputs', 'daptOutputs', 'cleanDaptReport']) {
    if (!/^[a-f0-9]{64}$/.test(contract.references?.[key] ?? '')) fail(`predeclared reference ${key} is missing`);
  }
}

// Stage 1: score the three already-generated full-suite outputs, build the
// accepted promotion table and create the blinded candidate-vs-DAPT pairs.
export async function prepareScreenplayQuality({
  contract, jobId, evalReport, checkpointSha256, suiteFile, scriptsDir = DEFAULT_SCREENPLAY_SCRIPTS_DIR,
  baseOutputs, daptOutputs, candidateOutputs, cleanDaptReport, cleanDaptModel,
  outDir,
}) {
  requireProfile(contract);
  if (!outDir || !jobId || !evalReport?.reportDigest || !evalReport.lineage?.tunedFingerprint || !checkpointSha256) fail('job/evaluation/checkpoint binding is incomplete');
  const files = { suite: suiteFile, baseOutputs, daptOutputs, candidateOutputs, cleanDaptReport };
  const sources = {};
  for (const [name, file] of Object.entries(files)) sources[name] = await source(file);
  if (sources.suite.sha256 !== contract.suiteSha256) fail('suite SHA-256 differs from the frozen contract');
  for (const [name, key] of [['baseOutputs', 'baseOutputs'], ['daptOutputs', 'daptOutputs'], ['cleanDaptReport', 'cleanDaptReport']]) {
    if (sources[name].sha256 !== contract.references[key]) fail(`${name} SHA-256 differs from the frozen reference`);
  }
  if (await screenplayScriptDigest(scriptsDir) !== contract.scorerSha256) fail('scorer script bundle differs from the frozen contract');
  const items = suiteItems(await readJson(suiteFile));
  const outputs = {};
  for (const name of ['baseOutputs', 'daptOutputs', 'candidateOutputs']) outputs[name] = await checkedOutputs(files[name], items);
  const root = path.resolve(outDir);
  await ensureDir(root);
  const resultsFile = path.join(root, 'qualitative-input.json');
  await writeJson(resultsFile, {
    prompts: items, prompts_json: path.resolve(suiteFile), seed: 20260824,
    models: [
      { name: 'base', outputs: outputs.baseOutputs },
      { name: 'dapt', outputs: outputs.daptOutputs },
      { name: 'candidate', outputs: outputs.candidateOutputs },
    ],
  });
  const scoreFile = path.join(root, 'generation-quality-scores.json');
  await python(path.join(scriptsDir, SCRIPT_NAMES.score), ['--results', resultsFile, '--suite', suiteFile, '--out-json', scoreFile, '--out-md', path.join(root, 'generation-quality-scores.md')]);
  const scored = await readJson(scoreFile);
  const dapt = scored.models?.find((model) => model.name === 'dapt');
  const daptRule = dapt?.aggregate?.rule_pass_rate;
  if (!finite(daptRule) || dapt.aggregate.items !== 120) fail('DAPT rule reference is missing or partial');
  const tableFile = path.join(root, 'promotion-table.json');
  await python(path.join(scriptsDir, SCRIPT_NAMES.table), [
    '--suite', suiteFile, '--scores', scoreFile, '--base-name', 'base', '--dapt-rule-pass', String(daptRule),
    '--out-json', tableFile, '--out-md', path.join(root, 'promotion-table.md'),
  ]);
  const blindDir = path.join(root, 'blind');
  await ensureDir(blindDir);
  await python(path.join(scriptsDir, SCRIPT_NAMES.blind), [
    'build', '--suite', suiteFile, '--reference', daptOutputs, '--candidate', candidateOutputs,
    '--reference-name', 'dapt', '--candidate-name', 'candidate', '--out-dir', blindDir, '--seed', '20260824',
  ]);
  const stage = {
    format: 'tmf.screenplay-quality-stage/v1', profile: SCREENPLAY_PROFILE, jobId,
    suiteSha256: contract.suiteSha256, scorerSha256: contract.scorerSha256,
    contractDigest: sha256Json(contract), checkpointSha256,
    evalReportDigest: evalReport.reportDigest, tunedFingerprint: evalReport.lineage.tunedFingerprint,
    cleanDaptModel, sources, scriptsDir: path.resolve(scriptsDir),
    scoreFile, scoreSha256: await sha256File(scoreFile), tableFile, tableSha256: await sha256File(tableFile),
    blindKeyFile: path.join(blindDir, 'key.json'), blindKeySha256: await sha256File(path.join(blindDir, 'key.json')),
  };
  stage.stageDigest = sha256Json({ ...stage, stageDigest: undefined });
  const stageFile = path.join(root, 'stage.json');
  await writeJson(stageFile, stage);
  return { stageFile, blindDir, scoreFile, tableFile, stageDigest: stage.stageDigest };
}

function loss(report, modelName) {
  const rows = report?.results;
  const row = rows?.find((item) => item.model === modelName && item.protocol === 'chat_wrapped');
  if (!row || !finite(row.token_weighted_loss) || row.token_weighted_loss <= 0 || row.samples < 64) fail(`clean-val chat_wrapped loss missing for ${modelName}`);
  return row.token_weighted_loss;
}

// Stage 2: require raw blinded verdicts from >=2 judges and independent
// clean-val loss before a report exists at all. Failed gates still produce a
// sealed report, but export remains blocked by the frozen contract.
export async function finalizeScreenplayQuality({ stageFile, contract, judgeVerdicts, cleanCandidateReport, cleanCandidateModel, reportFile }) {
  requireProfile(contract);
  const stage = await readJson(stageFile);
  if (stage.format !== 'tmf.screenplay-quality-stage/v1' || stage.stageDigest !== sha256Json({ ...stage, stageDigest: undefined }) || stage.contractDigest !== sha256Json(contract)) fail('stage integrity/contract mismatch');
  for (const ref of Object.values(stage.sources)) if (await sha256File(ref.file) !== ref.sha256) fail(`source changed after preparation: ${ref.file}`);
  if (await screenplayScriptDigest(stage.scriptsDir) !== stage.scorerSha256 || await sha256File(stage.scoreFile) !== stage.scoreSha256 ||
      await sha256File(stage.tableFile) !== stage.tableSha256 || await sha256File(stage.blindKeyFile) !== stage.blindKeySha256) fail('scorer or staged report changed after preparation');
  if (!Array.isArray(judgeVerdicts) || judgeVerdicts.length < 2) fail('at least two independent judge verdict files are required');
  const names = new Set();
  const specs = [];
  const judges = {};
  const ids = new Set(suiteItems(await readJson(stage.sources.suite.file)).map((item) => item.id));
  for (const spec of judgeVerdicts) {
    const at = spec.indexOf('=');
    if (at < 1) fail('judge verdict must be NAME=FILE');
    const name = spec.slice(0, at); const file = spec.slice(at + 1);
    if (!/^[a-zA-Z0-9_-]+$/.test(name) || names.has(name)) fail('judge names must be distinct safe identifiers');
    names.add(name);
    const rows = await readJsonl(file);
    if (rows.length !== 240 || new Set(rows.map((row) => `${row.order}:${row.id}`)).size !== 240 ||
      rows.some((row) => !ids.has(row.id) || !['order1', 'order2'].includes(row.order) || !['A', 'B', 'tie'].includes(row.verdict))) {
      fail(`${file}: judge must provide one verdict for each of 120 IDs in both orders`);
    }
    specs.push(`${name}=${path.resolve(file)}`);
    judges[name] = await source(file);
  }
  const tally = JSON.parse(await python(path.join(stage.scriptsDir, SCRIPT_NAMES.blind), [
    'tally', '--key', stage.blindKeyFile, '--suite', stage.sources.suite.file, '--verdicts', ...specs,
  ]));
  const cells = Object.values(tally.judges).flatMap((judge) => ['order1', 'order2'].map((order) => judge[order]));
  if (cells.length !== 2 * names.size || cells.some((cell) => !cell || cell.judged !== 120 || cell.expected !== 120)) fail('blind tally has incomplete cells');
  const daptLoss = await readJson(stage.sources.cleanDaptReport.file);
  const candidateLoss = await readJson(cleanCandidateReport);
  for (const field of ['val_jsonl', 'samples', 'max_length', 'quantization']) {
    if (candidateLoss[field] !== daptLoss[field]) fail(`clean-val ${field} protocol differs from DAPT reference`);
  }
  if (!candidateLoss.tokenizer || !candidateLoss.model?.class || !candidateLoss.model?.attn_implementation ||
      candidateLoss.base_model !== daptLoss.base_model ||
      sha256Json(candidateLoss.tokenizer) !== sha256Json(daptLoss.tokenizer) ||
      candidateLoss.model?.class !== daptLoss.model?.class ||
      candidateLoss.model?.attn_implementation !== daptLoss.model?.attn_implementation) {
    fail('clean-val base model, tokenizer or attention implementation differs from DAPT reference');
  }
  const referenceLoss = loss(daptLoss, stage.cleanDaptModel);
  const tunedLoss = loss(candidateLoss, cleanCandidateModel);
  const table = await readJson(stage.tableFile);
  const candidate = table.models?.candidate;
  const g = candidate?.gates;
  if (!g || g.rule_pass_gain_pt.reference !== 'dapt' || table.items !== 120) fail('promotion table is not a complete DAPT comparison');
  const broadCount = (candidate.by_tier.broad?.n ?? 0) + (candidate.by_tier.long?.n ?? 0);
  const metrics = { quality: {
    ruleGainPt: g.rule_pass_gain_pt.value,
    constrainedCount: candidate.overall.constrained_items,
    sceneCompleteRate: g.scene_complete_rate.value,
    sceneCount: candidate.by_tier.scene?.n ?? 0,
    broadEarlyEosRate: g.broad_early_eos_rate.value,
    broadEarlyEosDeltaVsBase: Number((g.broad_early_eos_rate.value - g.broad_early_eos_rate.base).toFixed(6)),
    broadCount,
    worstCategoryDropPt: g.major_category_drop_pt.worst,
    majorCategoryCount: table.major_task_types?.length ?? 0,
    abMinWinRate: Math.min(...cells.map((cell) => cell.candidate_win_rate)),
    abMinBootstrapLower: Math.min(...cells.map((cell) => cell.bootstrap95_lower)),
    abMinJudged: Math.min(...cells.map((cell) => cell.judged)),
    abJudgeCount: names.size,
    cleanValRegressionRelative: Number((tunedLoss / referenceLoss - 1).toFixed(6)),
  } };
  for (const value of Object.values(metrics.quality)) if (!finite(value)) fail('one or more quality metrics are missing/non-finite');
  const gates = evaluateGates(metrics, SCREENPLAY_GATES);
  const report = sealExternalQualityEvidence({
    format: EXTERNAL_QUALITY_FORMAT, profile: SCREENPLAY_PROFILE, jobId: stage.jobId,
    suiteSha256: stage.suiteSha256, scorerSha256: stage.scorerSha256,
    contractDigest: stage.contractDigest, checkpointSha256: stage.checkpointSha256,
    evalReportDigest: stage.evalReportDigest, tunedFingerprint: stage.tunedFingerprint,
    metrics, gates, evidence: {
      stageFile: await source(stageFile), scoreFile: { file: stage.scoreFile, sha256: stage.scoreSha256 },
      tableFile: { file: stage.tableFile, sha256: stage.tableSha256 }, blindKeyFile: { file: stage.blindKeyFile, sha256: stage.blindKeySha256 },
      judges, cleanCandidateReport: await source(cleanCandidateReport), cleanDaptReport: stage.sources.cleanDaptReport,
      outputs: { base: stage.sources.baseOutputs, dapt: stage.sources.daptOutputs, candidate: stage.sources.candidateOutputs },
    },
  });
  const dest = path.resolve(reportFile);
  await writeJson(dest, report);
  return { reportFile: dest, report };
}
