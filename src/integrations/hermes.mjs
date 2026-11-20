// Hermes adapter.
//
// Hermes is a messaging agent: a model picks a flat, prefixed tool name and
// hands the gateway a JSON *string* of arguments, and whatever comes back has
// to be text the model can read in the next turn. That is a different shape
// from the OpenClaw tool provider, but it must not be a different product, so
// this adapter reuses the action table in src/integrations/openclaw.mjs rather
// than restating it. Adding an action there surfaces it on both hosts.
//
// The contract this module promises a Hermes gateway:
//   * `toolSpecs()` is static and side-effect free, and returns OpenAI-style
//     function specs (`tmf_<action>`) with a closed JSON Schema per tool.
//   * `callTool(name, args)` always resolves to an envelope with `ok`, `tool`
//     and a string `content` the model can be shown — including on failure.
//   * `args` may be an object or the JSON string Hermes receives from the
//     model; unparsable JSON is `invalid-json`, not a crash.
//   * Unknown tools, unknown params and unauthorized compute are errors with
//     stable codes, mirroring the OpenClaw codes.

import { ACTION_SPECS, ACTION_NAMES, CONTRACT_VERSION } from './openclaw.mjs';
import * as api from '../api.mjs';
import { VERSION } from '../version.mjs';

export { CONTRACT_VERSION };
export const HOST = 'hermes';

/** Flat tool-name prefix; Hermes tool names are global to a gateway. */
export const TOOL_PREFIX = 'tmf_';

/** JSON Schema fragment per accepted param name, for the tool specs. */
const PARAM_SCHEMA = {
  configFile: { type: 'string', description: 'Path to a foundry config file; defaults to the adapter config.' },
  requireConfig: { type: 'boolean', description: 'Fail the check when no config file can be loaded.' },
  jobId: { type: 'string', description: 'Job id; omit to use the latest job in the workspace.' },
  auditTail: { type: 'integer', description: 'How many trailing audit-log lines to include.' },
  outDir: { type: 'string', description: 'Output directory.' },
  seed: { type: 'integer', description: 'Deterministic seed for synthesis.' },
  count: { type: 'integer', description: 'Number of synthetic examples to write.' },
  profile: { type: 'string', description: 'Synthetic corpus profile name.' },
  resume: { type: 'boolean', description: 'Resume an interrupted job instead of starting over.' },
  resumeCancelled: { type: 'boolean', description: 'Also resume a job that was cancelled.' },
  auditEveryStep: { type: 'integer', description: 'Audit-log cadence in training steps.' },
  gates: { type: 'object', description: 'Acceptance gate thresholds to apply to the eval report.' },
  reportFile: { type: 'string', description: 'Path to write the evaluation report to.' },
  maxNewTokens: { type: 'integer', description: 'Generation budget per eval sample.' },
  evalReport: { type: 'string', description: 'Path to an existing evaluation report to pin into the export.' },
  version: { type: 'string', description: 'Version label for the exported package.' },
  evaluateIfMissing: { type: 'boolean', description: 'Run evaluation first when no report is pinned.' },
};

export const TOOL_NAMES = Object.freeze(ACTION_NAMES.map((a) => `${TOOL_PREFIX}${a}`));

function actionOf(tool) {
  if (typeof tool !== 'string' || !tool.startsWith(TOOL_PREFIX)) return null;
  const action = tool.slice(TOOL_PREFIX.length);
  return Object.prototype.hasOwnProperty.call(ACTION_SPECS, action) ? action : null;
}

/** Static, side-effect-free function specs for a Hermes gateway. */
export function toolSpecs() {
  return ACTION_NAMES.map((action) => {
    const spec = ACTION_SPECS[action];
    const properties = {};
    for (const p of spec.params) properties[p] = PARAM_SCHEMA[p] ?? { description: `Parameter "${p}".` };
    return {
      type: 'function',
      function: {
        name: `${TOOL_PREFIX}${action}`,
        description: `${spec.summary}${spec.compute ? ' Consumes compute.' : ''}`,
        parameters: {
          type: 'object',
          properties,
          required: [...(spec.required ?? [])],
          additionalProperties: false,
        },
      },
    };
  });
}

/** Static description of the adapter, for `hermes`-side diagnostics. */
export function manifest() {
  return {
    host: HOST,
    product: api.PRODUCT,
    toolkitVersion: VERSION,
    contractVersion: CONTRACT_VERSION,
    transport: 'tool-call',
    toolPrefix: TOOL_PREFIX,
    tools: TOOL_NAMES.map((name) => ({
      name,
      action: name.slice(TOOL_PREFIX.length),
      compute: ACTION_SPECS[name.slice(TOOL_PREFIX.length)].compute,
    })),
  };
}

function textOf(value) {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function fail(tool, code, message) {
  return { ok: false, contractVersion: CONTRACT_VERSION, host: HOST, tool, error: { code, message }, content: `ERROR ${code}: ${message}` };
}

/**
 * Build an adapter bound to one workspace.
 *
 * `allowCompute: false` mounts the gateway read-only: the diagnostic tools
 * still answer, while anything that would burn compute or write artifacts is
 * refused with `compute-not-authorized` instead of running.
 */
export function createHermesAdapter({ baseDir = process.cwd(), configFile = null, allowCompute = true } = {}) {
  const ctx = { baseDir, configFile };

  async function callTool(tool, args = {}) {
    const action = actionOf(tool);
    if (!action) {
      return fail(typeof tool === 'string' ? tool : String(tool), 'unknown-tool', `unknown tool "${tool}"; known tools: ${TOOL_NAMES.join(', ')}`);
    }
    let params = args;
    if (typeof params === 'string') {
      const trimmed = params.trim();
      if (!trimmed) params = {};
      else {
        try {
          params = JSON.parse(trimmed);
        } catch (err) {
          return fail(tool, 'invalid-json', `arguments were not valid JSON: ${err?.message ?? String(err)}`);
        }
      }
    }
    if (params === null || params === undefined) params = {};
    if (typeof params !== 'object' || Array.isArray(params)) {
      return fail(tool, 'invalid-params', 'arguments must decode to a plain object');
    }
    const spec = ACTION_SPECS[action];
    const unknown = Object.keys(params).filter((k) => !spec.params.includes(k));
    if (unknown.length) {
      return fail(tool, 'unknown-params', `unknown params for "${tool}": ${unknown.join(', ')}`);
    }
    const missing = (spec.required ?? []).filter((k) => params[k] === undefined || params[k] === null);
    if (missing.length) {
      return fail(tool, 'missing-params', `tool "${tool}" requires: ${missing.join(', ')}`);
    }
    if (spec.compute && !allowCompute) {
      return fail(tool, 'compute-not-authorized', `tool "${tool}" consumes compute and this adapter was mounted read-only`);
    }
    try {
      const result = await spec.handler(ctx, params);
      return { ok: true, contractVersion: CONTRACT_VERSION, host: HOST, tool, action, result, content: textOf(result) };
    } catch (err) {
      const code = err?.code && typeof err.code === 'string' ? err.code : err?.name === 'ConfigError' ? 'config-error' : 'action-failed';
      return fail(tool, code, err?.message ?? String(err));
    }
  }

  return { host: HOST, baseDir, allowCompute, manifest, toolSpecs, tools: TOOL_NAMES, callTool };
}

/** Registration entry point for a Hermes gateway. */
export function register(host, options = {}) {
  const adapter = createHermesAdapter(options);
  const specs = toolSpecs();
  if (host && typeof host.registerTool === 'function') {
    for (const spec of specs) host.registerTool({ spec, handler: (args) => adapter.callTool(spec.function.name, args) });
  } else if (host && typeof host.registerTools === 'function') {
    host.registerTools({ tools: specs, handler: adapter.callTool });
  }
  return adapter;
}

export default { CONTRACT_VERSION, HOST, TOOL_PREFIX, TOOL_NAMES, manifest, toolSpecs, createHermesAdapter, register };
