// Deterministic, result-guided candidate proposals. The controller can create
// new hyperparameter configurations but cannot change data, model or gates.
import { validateConfig } from '../config.mjs';
import { sha256Json } from '../core/hash.mjs';

const ALLOWED_PATHS = new Set([
  'hyper.lr', 'hyper.batchSize', 'hyper.weightDecay', 'hyper.clipNorm',
  'hyper.warmupRatio', 'hyper.minLrRatio', 'hyper.lora.rank', 'hyper.lora.alpha',
]);
const INTEGER_PATHS = new Set(['hyper.batchSize', 'hyper.lora.rank']);
const MAX_DIMENSIONS = 4;
const MAX_COMBINATIONS = 64;

function plain(value) { return value && typeof value === 'object' && !Array.isArray(value); }
function atPath(value, dotted) { return dotted.split('.').reduce((obj, key) => obj?.[key], value); }
function setPath(value, dotted, next) {
  const keys = dotted.split('.');
  let obj = value;
  for (const key of keys.slice(0, -1)) obj = obj[key];
  obj[keys.at(-1)] = next;
}

export function buildProposalPool({ proposal, baseRaw, baseDir }) {
  if (!plain(proposal) || Object.keys(proposal).some((key) => !['baseConfigFile', 'strategy', 'parameters'].includes(key))) {
    throw new Error('campaign.proposal must contain only baseConfigFile, strategy and parameters');
  }
  if (proposal.strategy !== 'best-neighbor') throw new Error('campaign.proposal.strategy must be best-neighbor');
  if (!Array.isArray(proposal.parameters) || proposal.parameters.length < 1 || proposal.parameters.length > MAX_DIMENSIONS) {
    throw new Error(`campaign.proposal.parameters must contain 1-${MAX_DIMENSIONS} dimensions`);
  }
  const seen = new Set();
  const dimensions = proposal.parameters.map((entry, index) => {
    if (!plain(entry) || Object.keys(entry).some((key) => !['path', 'values'].includes(key))) {
      throw new Error(`campaign.proposal.parameters[${index}] must contain only path and values`);
    }
    if (!ALLOWED_PATHS.has(entry.path) || seen.has(entry.path)) throw new Error(`proposal path not allowed or duplicated: ${entry.path}`);
    seen.add(entry.path);
    const values = entry.values;
    if (!Array.isArray(values) || values.length < 2 || values.length > 16 ||
        values.some((value, i) => typeof value !== 'number' || !Number.isFinite(value) || value <= 0 ||
          (INTEGER_PATHS.has(entry.path) && !Number.isInteger(value)) || (i && value <= values[i - 1]))) {
      throw new Error(`${entry.path}: values must be 2-16 strictly increasing valid numbers`);
    }
    const baseValue = atPath(baseRaw, entry.path);
    const baselineIndex = values.indexOf(baseValue);
    if (baselineIndex < 0) throw new Error(`${entry.path}: base config value must appear in values`);
    return { path: entry.path, values, baselineIndex };
  });
  const count = dimensions.reduce((total, item) => total * item.values.length, 1);
  if (count > MAX_COMBINATIONS) throw new Error(`proposal grid has ${count} combinations, above ${MAX_COMBINATIONS}`);
  const pool = [];
  function visit(indices) {
    if (indices.length < dimensions.length) {
      for (let i = 0; i < dimensions[indices.length].values.length; i++) visit([...indices, i]);
      return;
    }
    const raw = structuredClone(baseRaw);
    const parameters = {};
    dimensions.forEach((dimension, index) => {
      const value = dimension.values[indices[index]];
      setPath(raw, dimension.path, value);
      parameters[dimension.path] = value;
    });
    const validated = validateConfig(raw, { baseDir });
    if (!validated.ok) throw new Error(`generated candidate invalid: ${validated.errors.map((item) => `${item.at}: ${item.message}`).join('; ')}`);
    pool.push({ id: `p-${indices.join('-')}`, indices, parameters, raw,
      config: validated.config, configSha256: sha256Json(raw), steps: validated.config.hyper.steps });
  }
  visit([]);
  return { pool, dimensions, baselineIndices: dimensions.map((item) => item.baselineIndex) };
}

function distance(left, right) { return left.reduce((sum, value, i) => sum + Math.abs(value - right[i]), 0); }
function byId(a, b) { return a.id < b.id ? -1 : a.id > b.id ? 1 : 0; }

export function chooseNextProposal({ pool, baselineIndices, previous, direction }) {
  const tried = new Set(previous.filter((item) => item?.id).map((item) => item.id));
  const scored = previous.filter((item) => item?.id && typeof item.score === 'number' && Number.isFinite(item.score));
  scored.sort((a, b) => (direction === 'min' ? a.score - b.score : b.score - a.score) || byId(a, b));
  const anchor = pool.find((item) => item.id === scored[0]?.id);
  const center = anchor?.indices ?? baselineIndices;
  const remaining = pool.filter((item) => !tried.has(item.id));
  remaining.sort((a, b) => distance(a.indices, center) - distance(b.indices, center) ||
    distance(a.indices, baselineIndices) - distance(b.indices, baselineIndices) ||
    byId(a, b));
  const chosen = remaining[0] ?? null;
  return chosen ? { ...chosen, parentId: anchor?.id ?? null } : null;
}
