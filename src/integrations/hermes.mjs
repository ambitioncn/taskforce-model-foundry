// Hermes adapter.
//
// Hermes is a messaging agent: a model picks a flat, prefixed tool name and
// hands the gateway a JSON *string* of arguments, and whatever comes back has
// to be text the model can read in the next turn. That is a different shape
// from the OpenClaw tool provider, but it must not be a different product, so
// this adapter consumes the harness-neutral operation contract. Adding an
// action there surfaces it on both hosts.
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

import { ACTION_SPECS, ACTION_NAMES, CONTRACT_VERSION, createOperationClient, parametersFor } from '../operations.mjs';
import * as api from '../api.mjs';
import { VERSION } from '../version.mjs';

export { CONTRACT_VERSION };
export const HOST = 'hermes';

/** Flat tool-name prefix; Hermes tool names are global to a gateway. */
export const TOOL_PREFIX = 'tmf_';

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
    return {
      type: 'function',
      function: {
        name: `${TOOL_PREFIX}${action}`,
        description: `${spec.summary}${spec.compute ? ' Consumes compute.' : ''}`,
        parameters: { ...parametersFor(action), required: [...(spec.required ?? [])] },
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
  const client = createOperationClient({ baseDir, configFile, allowCompute });

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
    const response = await client.invoke(action, params);
    if (!response.ok) return fail(tool, response.error.code, response.error.message);
    return { ...response, host: HOST, tool, content: textOf(response.result) };
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
