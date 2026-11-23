// Configuration schema and validation.
//
// The foundry runs long, expensive jobs, so a bad config must fail here --
// before a single training step -- with a message that names the exact path.
// Validation is therefore total: every field is checked, unknown keys are
// reported rather than silently ignored, and the returned config is the
// normalized object the rest of the product consumes.

import path from 'node:path';
import { readJson, exists } from './core/fsx.mjs';
import { COMPARATORS } from './eval/gates.mjs';

export const CONFIG_FORMAT = 'tmf-config-v1';
export const DEFAULT_CONFIG_FILES = ['tmf.config.json', '.tmfrc.json'];

export const CONFIG_DEFAULTS = {
  format: CONFIG_FORMAT,
  project: 'tmf-project',
  runRoot: '.tmf-run',
  backend: 'builtin-js',
  mode: 'lora',
  seed: 'tmf-train',
  model: { maxVocab: 256, maxSeqLen: 96 },
  data: { datasets: [], config: {} },
  hyper: {},
  budget: {},
  trainBudget: {},
  eval: {},
};

const TOP_LEVEL = new Set([
  'format', 'project', 'runRoot', 'backend', 'mode', 'seed',
  'model', 'data', 'hyper', 'budget', 'trainBudget', 'eval',
]);
// Mode names must match what a trainer backend advertises in
// capabilities.modes; anything else can never pass doctor's mode check.
export const MODES = new Set(['lora', 'pretrain', 'full-finetune']);

class Errors {
  constructor() { this.list = []; }
  add(at, message) { this.list.push({ at, message }); }
  get ok() { return this.list.length === 0; }
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireString(errors, at, value, fallback) {
  if (value === undefined) return fallback;
  if (typeof value !== 'string' || value.trim() === '') {
    errors.add(at, 'must be a non-empty string');
    return fallback;
  }
  return value;
}

function requirePositiveInt(errors, at, value, fallback) {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value <= 0) {
    errors.add(at, 'must be a positive integer');
    return fallback;
  }
  return value;
}

function requireObject(errors, at, value, fallback) {
  if (value === undefined) return fallback;
  if (!isPlainObject(value)) {
    errors.add(at, 'must be an object');
    return fallback;
  }
  return value;
}

function validateDatasets(errors, raw) {
  if (!Array.isArray(raw)) {
    errors.add('data.datasets', 'must be an array of { name, path } entries');
    return [];
  }
  const seen = new Set();
  return raw.map((entry, i) => {
    const at = `data.datasets[${i}]`;
    if (!isPlainObject(entry)) {
      errors.add(at, 'must be an object with name and path');
      return null;
    }
    const name = requireString(errors, `${at}.name`, entry.name, null);
    const file = requireString(errors, `${at}.path`, entry.path, null);
    if (name === null) errors.add(`${at}.name`, 'is required');
    if (file === null) errors.add(`${at}.path`, 'is required');
    if (name && seen.has(name)) errors.add(`${at}.name`, `duplicate dataset name "${name}"`);
    if (name) seen.add(name);
    const out = { name, path: file };
    if (entry.format !== undefined) out.format = requireString(errors, `${at}.format`, entry.format, 'jsonl');
    if (entry.license !== undefined) out.license = requireString(errors, `${at}.license`, entry.license, null);
    if (entry.source !== undefined) out.source = requireString(errors, `${at}.source`, entry.source, null);
    return out;
  }).filter(Boolean);
}

function validateBudget(errors, at, raw) {
  const budget = requireObject(errors, at, raw, {});
  const out = {};
  for (const [key, value] of Object.entries(budget)) {
    if (!['maxTokens', 'maxSteps', 'maxWallClockMs', 'maxCostCny'].includes(key)) {
      errors.add(`${at}.${key}`, 'unknown budget key (allowed: maxTokens, maxSteps, maxWallClockMs, maxCostCny)');
      continue;
    }
    if (typeof value !== 'number' || !(value > 0)) {
      errors.add(`${at}.${key}`, 'must be a number greater than 0');
      continue;
    }
    out[key] = value;
  }
  return out;
}

// Gate thresholds are project policy, not a toolkit constant: what counts as
// an acceptable holdout loss depends on the model and the task, and the
// built-in defaults were tuned against the builtin-js demo. They stayed
// unreachable until now -- api.mjs already read `eval.gates`, but `eval` was
// not an allowed top-level key, so no config could ever set it. Declaring a
// gate list replaces the defaults wholesale rather than merging, because a
// half-overridden gate set is the kind of thing that silently drops a check.
const GATE_KEYS = new Set(['id', 'metric', 'comparator', 'threshold', 'sampleMetric', 'minSamples', 'description']);

function validateGate(errors, at, raw) {
  if (!isPlainObject(raw)) {
    errors.add(at, 'must be an object');
    return null;
  }
  for (const key of Object.keys(raw)) {
    if (!GATE_KEYS.has(key)) errors.add(`${at}.${key}`, `unknown gate key (allowed: ${[...GATE_KEYS].join(', ')})`);
  }
  const gate = {
    id: requireString(errors, `${at}.id`, raw.id, ''),
    metric: requireString(errors, `${at}.metric`, raw.metric, ''),
    comparator: requireString(errors, `${at}.comparator`, raw.comparator, ''),
  };
  if (gate.comparator && !COMPARATORS[gate.comparator]) {
    errors.add(`${at}.comparator`, `must be one of ${Object.keys(COMPARATORS).join(', ')}`);
  }
  if (typeof raw.threshold !== 'number' || Number.isNaN(raw.threshold)) {
    errors.add(`${at}.threshold`, 'must be a number');
  } else {
    gate.threshold = raw.threshold;
  }
  if (raw.sampleMetric !== undefined) gate.sampleMetric = requireString(errors, `${at}.sampleMetric`, raw.sampleMetric, '');
  if (raw.minSamples !== undefined) gate.minSamples = requirePositiveInt(errors, `${at}.minSamples`, raw.minSamples, 1);
  if (raw.minSamples !== undefined && raw.sampleMetric === undefined) {
    errors.add(`${at}.sampleMetric`, 'is required when minSamples is set, otherwise the sample floor cannot be read');
  }
  if (raw.description !== undefined) gate.description = requireString(errors, `${at}.description`, raw.description, '');
  return gate;
}

function validateEval(errors, at, raw) {
  const evalRaw = requireObject(errors, at, raw, {});
  for (const key of Object.keys(evalRaw)) {
    if (!['gates', 'externalQuality'].includes(key)) errors.add(`${at}.${key}`, 'unknown eval key (allowed: gates, externalQuality)');
  }
  const result = {};
  if (evalRaw.externalQuality !== undefined) {
    const quality = requireObject(errors, `${at}.externalQuality`, evalRaw.externalQuality, {});
    for (const key of Object.keys(quality)) {
      if (!['suiteSha256', 'scorerSha256', 'gates', 'profile', 'references'].includes(key)) errors.add(`${at}.externalQuality.${key}`, 'unknown external quality key');
    }
    for (const key of ['suiteSha256', 'scorerSha256']) {
      if (typeof quality[key] !== 'string' || !/^[a-f0-9]{64}$/.test(quality[key])) errors.add(`${at}.externalQuality.${key}`, 'must be a lowercase SHA-256 digest');
    }
    if (!Array.isArray(quality.gates) || quality.gates.length === 0) {
      errors.add(`${at}.externalQuality.gates`, 'must contain at least one predeclared gate');
    } else {
      const gates = quality.gates.map((gate, i) => validateGate(errors, `${at}.externalQuality.gates[${i}]`, gate)).filter(Boolean);
      if (new Set(gates.map((gate) => gate.id)).size !== gates.length) errors.add(`${at}.externalQuality.gates`, 'duplicate gate id');
      result.externalQuality = { suiteSha256: quality.suiteSha256, scorerSha256: quality.scorerSha256, gates };
      if (quality.profile !== undefined) result.externalQuality.profile = requireString(errors, `${at}.externalQuality.profile`, quality.profile, null);
      if (quality.references !== undefined) {
        const refs = requireObject(errors, `${at}.externalQuality.references`, quality.references, {});
        for (const [name, digest] of Object.entries(refs)) {
          if (!/^[a-zA-Z][a-zA-Z0-9]*$/.test(name) || typeof digest !== 'string' || !/^[a-f0-9]{64}$/.test(digest)) {
            errors.add(`${at}.externalQuality.references.${name}`, 'must be a named lowercase SHA-256 digest');
          }
        }
        result.externalQuality.references = refs;
      }
    }
  }
  if (evalRaw.gates === undefined) return result;
  if (!Array.isArray(evalRaw.gates)) {
    errors.add(`${at}.gates`, 'must be an array of gate objects');
    return result;
  }
  if (evalRaw.gates.length === 0) {
    errors.add(`${at}.gates`, 'must declare at least one gate; omit `eval.gates` to use the built-in defaults');
    return result;
  }
  const gates = evalRaw.gates.map((gate, i) => validateGate(errors, `${at}.gates[${i}]`, gate)).filter(Boolean);
  const seen = new Set();
  for (const gate of gates) {
    if (!gate.id) continue;
    if (seen.has(gate.id)) errors.add(`${at}.gates`, `duplicate gate id "${gate.id}"`);
    seen.add(gate.id);
  }
  return { ...result, gates };
}

// Hyperparameters are a whitelist rather than a free-form bag: an unknown key
// here used to be accepted and then silently dropped by the trainer, which is
// how a misspelled `steps` could look like it had been applied.
const HYPER_NUMBER_KEYS = ['steps', 'lr', 'batchSize', 'beta1', 'beta2', 'eps', 'weightDecay', 'clipNorm', 'warmupRatio', 'minLrRatio'];
const HYPER_OBJECT_KEYS = ['lora'];
const HYPER_KEYS = [...HYPER_NUMBER_KEYS, ...HYPER_OBJECT_KEYS];

function validateHyper(errors, at, raw) {
  const hyper = requireObject(errors, at, raw, {});
  const out = {};
  for (const [key, value] of Object.entries(hyper)) {
    if (!HYPER_KEYS.includes(key)) {
      errors.add(`${at}.${key}`, `unknown hyper key (allowed: ${HYPER_KEYS.join(', ')})`);
      continue;
    }
    if (HYPER_OBJECT_KEYS.includes(key)) {
      if (!isPlainObject(value)) {
        errors.add(`${at}.${key}`, 'must be an object');
        continue;
      }
      out[key] = value;
      continue;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      errors.add(`${at}.${key}`, 'must be a finite number greater than 0');
      continue;
    }
    if ((key === 'steps' || key === 'batchSize') && !Number.isInteger(value)) {
      errors.add(`${at}.${key}`, 'must be an integer');
      continue;
    }
    out[key] = value;
  }
  return out;
}

const HF_OPTIONS = new Set([
  'loadIn4Bit', 'quantType', 'doubleQuant', 'dtype', 'gradientCheckpointing',
  'gradientAccumulationSteps', 'lrScheduler', 'warmupSteps', 'optimizer', 'maxGradNorm',
  'maxMemoryMb', 'evalMaxSeconds',
]);

function validateHfOptions(errors, raw) {
  const hf = requireObject(errors, 'model.hf', raw, {});
  for (const [key, value] of Object.entries(hf)) {
    if (!HF_OPTIONS.has(key)) {
      errors.add(`model.hf.${key}`, 'unknown HF option');
    } else if (['loadIn4Bit', 'doubleQuant', 'gradientCheckpointing'].includes(key) && typeof value !== 'boolean') {
      errors.add(`model.hf.${key}`, 'must be boolean');
    } else if (['quantType', 'dtype', 'lrScheduler', 'optimizer'].includes(key) &&
      (typeof value !== 'string' || !value.trim())) {
      errors.add(`model.hf.${key}`, 'must be a non-empty string');
    } else if (['gradientAccumulationSteps', 'warmupSteps', 'maxMemoryMb', 'evalMaxSeconds'].includes(key) &&
      (!Number.isInteger(value) || value < (key === 'warmupSteps' ? 0 : 1))) {
      errors.add(`model.hf.${key}`, 'must be a non-negative or positive integer');
    } else if (key === 'maxGradNorm' && (typeof value !== 'number' || value <= 0)) {
      errors.add(`model.hf.${key}`, 'must be a positive number');
    }
  }
  if (hf.quantType !== undefined && hf.quantType !== 'nf4') errors.add('model.hf.quantType', 'only nf4 is supported');
  if (hf.dtype !== undefined && !['float32', 'bfloat16'].includes(hf.dtype)) {
    errors.add('model.hf.dtype', 'must be float32 or bfloat16');
  }
  return hf;
}

/**
 * Validate and normalize a raw config object.
 *
 * Returns { ok, config, errors } -- never throws on bad input, so callers can
 * render every problem at once instead of one per run.
 */
export function validateConfig(raw, { baseDir = process.cwd() } = {}) {
  const errors = new Errors();
  if (!isPlainObject(raw)) {
    errors.add('<root>', 'config must be a JSON object');
    return { ok: false, config: null, errors: errors.list };
  }

  for (const key of Object.keys(raw)) {
    if (!TOP_LEVEL.has(key)) errors.add(key, `unknown config key (allowed: ${[...TOP_LEVEL].join(', ')})`);
  }

  const format = requireString(errors, 'format', raw.format, CONFIG_FORMAT);
  if (format !== CONFIG_FORMAT) errors.add('format', `must be "${CONFIG_FORMAT}"`);

  const mode = requireString(errors, 'mode', raw.mode, CONFIG_DEFAULTS.mode);
  if (!MODES.has(mode)) errors.add('mode', `must be one of ${[...MODES].join(', ')}`);

  const modelRaw = requireObject(errors, 'model', raw.model, {});
  const model = {
    ...modelRaw,
    maxVocab: requirePositiveInt(errors, 'model.maxVocab', modelRaw.maxVocab, CONFIG_DEFAULTS.model.maxVocab),
    maxSeqLen: requirePositiveInt(errors, 'model.maxSeqLen', modelRaw.maxSeqLen, CONFIG_DEFAULTS.model.maxSeqLen),
  };
  if (modelRaw.hf !== undefined) model.hf = validateHfOptions(errors, modelRaw.hf);

  const dataRaw = requireObject(errors, 'data', raw.data, {});
  const datasets = validateDatasets(errors, dataRaw.datasets ?? []);
  if (datasets.length === 0 && Array.isArray(dataRaw.datasets)) {
    errors.add('data.datasets', 'at least one dataset is required to train');
  }

  const runRoot = requireString(errors, 'runRoot', raw.runRoot, CONFIG_DEFAULTS.runRoot);

  const config = {
    format: CONFIG_FORMAT,
    project: requireString(errors, 'project', raw.project, CONFIG_DEFAULTS.project),
    runRoot,
    runRootAbsolute: path.resolve(baseDir, runRoot),
    baseDir: path.resolve(baseDir),
    backend: requireString(errors, 'backend', raw.backend, CONFIG_DEFAULTS.backend),
    mode,
    seed: requireString(errors, 'seed', raw.seed, CONFIG_DEFAULTS.seed),
    model,
    data: { datasets, config: requireObject(errors, 'data.config', dataRaw.config, {}) },
    hyper: validateHyper(errors, 'hyper', raw.hyper),
    budget: validateBudget(errors, 'budget', raw.budget),
    trainBudget: requireObject(errors, 'trainBudget', raw.trainBudget, {}),
    eval: validateEval(errors, 'eval', raw.eval),
  };

  return { ok: errors.ok, config: errors.ok ? config : null, errors: errors.list };
}

export function formatConfigErrors(errors) {
  return errors.map((e) => `  - ${e.at}: ${e.message}`).join('\n');
}

export class ConfigError extends Error {
  constructor(message, errors = []) {
    super(message);
    this.name = 'ConfigError';
    this.code = 'TMF_CONFIG_INVALID';
    this.errors = errors;
  }
}

export async function findConfigFile(baseDir = process.cwd()) {
  for (const name of DEFAULT_CONFIG_FILES) {
    const file = path.join(baseDir, name);
    if (await exists(file)) return file;
  }
  return null;
}

/**
 * Load, validate and normalize a config file. Throws ConfigError carrying the
 * full error list so a CLI can print all problems and exit non-zero.
 */
export async function loadConfig({ file = null, baseDir = process.cwd() } = {}) {
  const resolved = file ? path.resolve(baseDir, file) : await findConfigFile(baseDir);
  if (!resolved) {
    throw new ConfigError(
      `no config file found in ${baseDir} (looked for ${DEFAULT_CONFIG_FILES.join(', ')})`,
      [{ at: '<file>', message: 'missing config file; run `tmf init` to create one' }],
    );
  }
  if (!(await exists(resolved))) {
    throw new ConfigError(`config file not found: ${resolved}`, [{ at: '<file>', message: 'missing config file' }]);
  }
  let raw;
  try {
    raw = await readJson(resolved);
  } catch (err) {
    throw new ConfigError(`config file is not valid JSON: ${resolved}`, [{ at: '<file>', message: err.message }]);
  }
  const { ok, config, errors } = validateConfig(raw, { baseDir: path.dirname(resolved) });
  if (!ok) throw new ConfigError(`invalid config: ${resolved}`, errors);
  return { file: resolved, config };
}

export function exampleConfig({ project = 'tmf-project', dataDir = 'data' } = {}) {
  return {
    format: CONFIG_FORMAT,
    project,
    runRoot: '.tmf-run',
    backend: 'builtin-js',
    mode: 'lora',
    seed: 'tmf-train',
    model: { maxVocab: 256, maxSeqLen: 96 },
    data: {
      datasets: [{ name: 'tasks', path: `${dataDir}/tasks.jsonl`, format: 'jsonl', license: 'internal', source: 'synthetic' }],
      config: {},
    },
    // lr stays at the small end on purpose: 0.05 oscillates on the builtin-js
    // AdamW path and loses the holdout improvement, while 0.005 converged
    // (holdout loss 0.248876 vs 4.82943 baseline on the reference run).
    hyper: { steps: 120, lr: 0.005, batchSize: 4 },
    budget: { maxTokens: 2000000, maxWallClockMs: 900000 },
  };
}
