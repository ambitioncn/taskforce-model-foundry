// OpenClaw host plugin entry.
//
// The host discovers this file through `openclaw.extensions` in package.json
// plus the `openclaw.plugin.json` manifest at the package root. It is a thin
// wrapper: the tool list comes from ./tools.mjs, which forwards every call to
// the in-process adapter in src/integrations/openclaw.mjs. The action names and
// parameter types come from the harness-neutral src/operations.mjs contract.
//
// Naming: adapter action `train` is published as tool `tmf_train`, matching
// `contracts.tools` in openclaw.plugin.json.

import { defineToolPlugin } from 'openclaw/plugin-sdk/tool-plugin';

import { CONTRACT_VERSION } from '../../src/integrations/openclaw.mjs';
import { buildTools } from './tools.mjs';

export { PARAM_SCHEMA, TOOL_PREFIX, buildTools, parametersFor, toolNameFor } from './tools.mjs';

export const PLUGIN_ID = 'taskforce-model-foundry';

export default defineToolPlugin({
  id: PLUGIN_ID,
  name: 'Taskforce Model Foundry',
  description: `Taskforce Model Foundry tools (contract v${CONTRACT_VERSION}).`,
  tools: (tool, ctx) => buildTools(tool, { config: ctx?.config ?? {} }),
});
