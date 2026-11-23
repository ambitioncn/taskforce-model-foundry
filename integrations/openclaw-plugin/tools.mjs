// Tool surface for the OpenClaw host plugin, kept free of any `openclaw`
// import so the package's own test suite can assert it without the host
// package installed. index.mjs adds nothing but the host registration.

import { createOpenClawAdapter } from '../../src/integrations/openclaw.mjs';

import { ACTION_SPECS, PARAM_SCHEMA, parametersFor } from '../../src/operations.mjs';
export { PARAM_SCHEMA, parametersFor } from '../../src/operations.mjs';

export const TOOL_PREFIX = 'tmf_';

/** `tmf_train` -> `train`; exported so the host test can assert the mapping. */
export function toolNameFor(action) {
  return `${TOOL_PREFIX}${action}`;
}

/**
 * Build the tool list from the adapter's own action table so adding an action
 * in the adapter publishes it here too.
 */
export function buildTools(tool, { config = {} } = {}) {
  const adapter = createOpenClawAdapter({
    baseDir: config.baseDir ?? process.cwd(),
    configFile: config.configFile ?? null,
    allowCompute: config.allowCompute ?? true,
  });
  return Object.keys(ACTION_SPECS).map((action) =>
    tool({
      name: toolNameFor(action),
      description: ACTION_SPECS[action].summary,
      parameters: parametersFor(action),
      execute: async (params) => adapter.invoke(action, params ?? {}),
    }),
  );
}

