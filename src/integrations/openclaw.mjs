// OpenClaw adapter.
//
// OpenClaw hosts load foundry capability as an in-process tool provider: the
// host passes an action name plus a JSON params object and expects a
// JSON-serializable envelope back, never a thrown exception and never a shell
// out. Everything here therefore goes through src/api.mjs, so the adapter and
// the `tmf` CLI cannot drift apart.
//
// The contract this module promises to an OpenClaw host:
//   * `manifest()` is static, side-effect free, and declares every action with
//     its accepted params and whether invoking it consumes compute.
//   * `invoke(action, params)` always resolves to
//     `{ ok, contractVersion, action, result }` or
//     `{ ok: false, contractVersion, action, error: { code, message } }`.
//   * Unknown actions, unknown params and unauthorized compute are errors with
//     stable codes, not crashes.
//   * No config value is echoed back verbatim; results are the API records.

import * as api from '../api.mjs';
import { startBackgroundEvaluation, backgroundEvaluationStatus } from './async-evaluate.mjs';
import { VERSION } from '../version.mjs';

/** Bumped only on a breaking change to the action set or envelope shape. */
export const CONTRACT_VERSION = 1;
export const HOST = 'openclaw';

/**
 * Action table. `compute: true` marks actions that spend real CPU/GPU time and
 * write run artifacts, which a host may gate behind its own authority check.
 */
const ACTIONS = {
  doctor: {
    summary: 'Environment and configuration self-check; reports every check.',
    params: ['configFile', 'requireConfig'],
    compute: false,
    handler: (ctx, p) => api.doctor({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, requireConfig: p.requireConfig ?? false }),
  },
  status: {
    summary: 'Durable job state plus a tail of its audit log.',
    params: ['jobId', 'configFile', 'auditTail'],
    compute: false,
    required: ['jobId'],
    handler: (ctx, p) => api.status({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId, auditTail: p.auditTail ?? 10 }),
  },
  synthesize: {
    summary: 'Write a synthetic training corpus (no private data involved).',
    params: ['outDir', 'seed', 'count', 'profile'],
    compute: true,
    handler: (ctx, p) => api.synthesize({ baseDir: ctx.baseDir, ...p }),
  },
  train: {
    summary: 'Run a post-training job that updates real weights or an adapter.',
    params: ['configFile', 'jobId', 'resume', 'resumeCancelled', 'auditEveryStep'],
    compute: true,
    handler: (ctx, p) => api.train({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId ?? null, resume: p.resume ?? true, resumeCancelled: p.resumeCancelled ?? false, auditEveryStep: p.auditEveryStep ?? 10 }),
  },
  evaluate: {
    summary: 'Score base vs trained model on the independent holdout and apply gates.',
    params: ['configFile', 'jobId', 'gates', 'reportFile', 'maxNewTokens', 'background', 'retry'],
    compute: true,
    handler: (ctx, p) => p.background
      ? startBackgroundEvaluation({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId, gates: p.gates ?? null, reportFile: p.reportFile ?? null, maxNewTokens: p.maxNewTokens ?? 48, retry: p.retry ?? false })
      : api.evaluate({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId ?? null, gates: p.gates ?? null, reportFile: p.reportFile ?? null, maxNewTokens: p.maxNewTokens ?? 48 }),
  },
  evaluate_status: {
    summary: 'Read a durable background evaluation receipt without consuming compute.',
    params: ['configFile', 'jobId', 'requestId'],
    required: ['jobId', 'requestId'],
    compute: false,
    handler: (ctx, p) => backgroundEvaluationStatus({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId, requestId: p.requestId }),
  },
  export: {
    summary: 'Package weights/adapter with lineage, model card and checksums.',
    params: ['configFile', 'jobId', 'evalReport', 'outDir', 'version', 'evaluateIfMissing'],
    compute: true,
    handler: (ctx, p) => api.exportModel({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId ?? null, evalReport: p.evalReport ?? null, outDir: p.outDir ?? null, version: p.version ?? null, evaluateIfMissing: p.evaluateIfMissing ?? true }),
  },
};

export const ACTION_NAMES = Object.freeze(Object.keys(ACTIONS));

/**
 * The action table itself, shared with the other host adapters (see
 * src/integrations/hermes.mjs) so a second host cannot drift from this one:
 * one action set, one set of handlers, one place to add an action.
 */
export const ACTION_SPECS = Object.freeze(ACTIONS);

/** Static, side-effect-free description of what this adapter offers a host. */
export function manifest() {
  return {
    host: HOST,
    product: api.PRODUCT,
    toolkitVersion: VERSION,
    contractVersion: CONTRACT_VERSION,
    transport: 'in-process',
    actions: ACTION_NAMES.map((name) => ({
      name,
      summary: ACTIONS[name].summary,
      params: [...ACTIONS[name].params],
      required: [...(ACTIONS[name].required ?? [])],
      compute: ACTIONS[name].compute,
    })),
  };
}

function fail(action, code, message) {
  return { ok: false, contractVersion: CONTRACT_VERSION, action, error: { code, message } };
}

/**
 * Build an adapter bound to one workspace.
 *
 * `allowCompute: false` lets a host mount the adapter for inspection only —
 * doctor and status still answer, while anything that would burn compute or
 * write artifacts is refused with `compute-not-authorized` instead of running.
 */
export function createOpenClawAdapter({ baseDir = process.cwd(), configFile = null, allowCompute = true } = {}) {
  const ctx = { baseDir, configFile };

  async function invoke(action, params = {}) {
    if (typeof action !== 'string' || !action) {
      return fail(String(action), 'invalid-action', 'action must be a non-empty string');
    }
    const spec = ACTIONS[action];
    if (!spec) {
      return fail(action, 'unknown-action', `unknown action "${action}"; known actions: ${ACTION_NAMES.join(', ')}`);
    }
    if (params === null || typeof params !== 'object' || Array.isArray(params)) {
      return fail(action, 'invalid-params', 'params must be a plain object');
    }
    const unknown = Object.keys(params).filter((k) => !spec.params.includes(k));
    if (unknown.length) {
      return fail(action, 'unknown-params', `unknown params for "${action}": ${unknown.join(', ')}`);
    }
    const missing = (spec.required ?? []).filter((k) => params[k] === undefined || params[k] === null);
    if (missing.length) {
      return fail(action, 'missing-params', `action "${action}" requires: ${missing.join(', ')}`);
    }
    if (spec.compute && !allowCompute) {
      return fail(action, 'compute-not-authorized', `action "${action}" consumes compute and this adapter was mounted read-only`);
    }
    try {
      const result = await spec.handler(ctx, params);
      return { ok: true, contractVersion: CONTRACT_VERSION, action, result };
    } catch (err) {
      const code = err?.code && typeof err.code === 'string' ? err.code : err?.name === 'ConfigError' ? 'config-error' : 'action-failed';
      return fail(action, code, err?.message ?? String(err));
    }
  }

  return { host: HOST, baseDir, allowCompute, manifest, actions: ACTION_NAMES, invoke };
}

/** Registration entry point for an OpenClaw plugin host. */
export function register(host, options = {}) {
  const adapter = createOpenClawAdapter(options);
  if (host && typeof host.registerToolProvider === 'function') {
    host.registerToolProvider({ name: api.PRODUCT, manifest: manifest(), invoke: adapter.invoke });
  }
  return adapter;
}

export default { CONTRACT_VERSION, HOST, manifest, createOpenClawAdapter, register, ACTION_NAMES };
