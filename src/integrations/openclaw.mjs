// OpenClaw adapter: registration and host metadata only. Operations live in
// the harness-neutral contract and remain available without OpenClaw installed.
import * as api from '../api.mjs';
import { VERSION } from '../version.mjs';
import { ACTION_NAMES, ACTION_SPECS, CONTRACT_VERSION, createOperationClient } from '../operations.mjs';

export { ACTION_NAMES, ACTION_SPECS, CONTRACT_VERSION } from '../operations.mjs';
export const HOST = 'openclaw';

export function manifest() {
  return {
    host: HOST, product: api.PRODUCT, toolkitVersion: VERSION,
    contractVersion: CONTRACT_VERSION, transport: 'in-process',
    actions: ACTION_NAMES.map((name) => ({
      name, summary: ACTION_SPECS[name].summary,
      params: [...ACTION_SPECS[name].params],
      required: [...(ACTION_SPECS[name].required ?? [])],
      compute: ACTION_SPECS[name].compute,
    })),
  };
}

export function createOpenClawAdapter(options = {}) {
  const client = createOperationClient(options);
  return { ...client, host: HOST, manifest };
}

export function register(host, options = {}) {
  const adapter = createOpenClawAdapter(options);
  if (host && typeof host.registerToolProvider === 'function') {
    host.registerToolProvider({ name: api.PRODUCT, manifest: manifest(), invoke: adapter.invoke });
  }
  return adapter;
}

export default { CONTRACT_VERSION, HOST, manifest, createOpenClawAdapter, register, ACTION_NAMES };
