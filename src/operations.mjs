// Harness-neutral operation contract and dispatch. CLI, OpenClaw, Hermes and
// future hosts must use the same actions, typed parameters, authority flag and
// response envelope. This module imports no agent harness package.
import * as api from './api.mjs';
import path from 'node:path';
import { startBackgroundEvaluation, backgroundEvaluationStatus } from './jobs/async-evaluate.mjs';
import { startBackgroundTraining, backgroundTrainingStatus, cancelBackgroundTraining } from './jobs/async-train.mjs';
import { startCampaign, backgroundCampaignStatus, cancelCampaign } from './controller/background.mjs';

export const CONTRACT_VERSION = 1;
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
    params: ['configFile', 'jobId', 'resume', 'resumeCancelled', 'auditEveryStep', 'background', 'retry'],
    compute: true,
    handler: (ctx, p) => {
      if (p.retry && !p.background) throw new Error('train retry requires background=true');
      if (p.background && p.resume === false) throw new Error('background training requires resume=true');
      if (p.background && !p.jobId) throw new Error('background training requires jobId');
      return p.background
        ? startBackgroundTraining({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId, resumeCancelled: p.resumeCancelled ?? false, auditEveryStep: p.auditEveryStep ?? 10, retry: p.retry ?? false })
        : api.train({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId ?? null, resume: p.resume ?? true, resumeCancelled: p.resumeCancelled ?? false, auditEveryStep: p.auditEveryStep ?? 10 });
    },
  },
  train_status: {
    summary: 'Read a durable background training receipt without consuming compute.',
    params: ['configFile', 'jobId', 'requestId'],
    required: ['jobId', 'requestId'],
    compute: false,
    handler: (ctx, p) => backgroundTrainingStatus({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId, requestId: p.requestId }),
  },
  train_cancel: {
    summary: 'Cancel a background training request while preserving completed checkpoints.',
    params: ['configFile', 'jobId', 'requestId'],
    required: ['jobId', 'requestId'],
    compute: true,
    handler: (ctx, p) => cancelBackgroundTraining({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId, requestId: p.requestId }),
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
    params: ['configFile', 'jobId', 'evalReport', 'qualityReport', 'outDir', 'version', 'evaluateIfMissing'],
    compute: true,
    handler: (ctx, p) => api.exportModel({ baseDir: ctx.baseDir, configFile: p.configFile ?? ctx.configFile ?? null, jobId: p.jobId ?? null, evalReport: p.evalReport ?? null, qualityReport: p.qualityReport ?? null, outDir: p.outDir ?? null, version: p.version ?? null, evaluateIfMissing: p.evaluateIfMissing ?? true }),
  },
  campaign_start: {
    summary: 'Start a bounded preapproved training campaign in an independent worker.',
    params: ['planFile', 'resumeCancelled'], required: ['planFile'], compute: true,
    handler: (ctx, p) => startCampaign({ planFile: path.resolve(ctx.baseDir, p.planFile), resumeCancelled: p.resumeCancelled ?? false }),
  },
  campaign_status: {
    summary: 'Read a durable campaign decision and worker status.',
    params: ['planFile'], required: ['planFile'], compute: false,
    handler: (ctx, p) => backgroundCampaignStatus({ planFile: path.resolve(ctx.baseDir, p.planFile) }),
  },
  campaign_cancel: {
    summary: 'Request cancellation of an active campaign worker.',
    params: ['planFile'], required: ['planFile'], compute: true,
    handler: (ctx, p) => cancelCampaign({ planFile: path.resolve(ctx.baseDir, p.planFile) }),
  },
};


export const ACTION_NAMES = Object.freeze(Object.keys(ACTIONS));
export const ACTION_SPECS = Object.freeze(ACTIONS);

// The single JSON Schema source for every host's tool definitions.
export const PARAM_SCHEMA = Object.freeze({
  configFile: { type: 'string', description: 'Foundry config file path.' },
  planFile: { type: 'string', description: 'Preapproved campaign manifest path.' },
  requireConfig: { type: 'boolean', description: 'Fail when no config file is present.' },
  jobId: { type: 'string', description: 'Durable Foundry job id.' },
  auditTail: { type: 'integer', description: 'How many audit-log lines to return.' },
  outDir: { type: 'string', description: 'Output directory.' },
  seed: { type: 'integer', description: 'Deterministic seed.' },
  count: { type: 'integer', description: 'Number of synthetic records to write.' },
  profile: { type: 'string', description: 'Synthetic corpus profile name.' },
  resume: { type: 'boolean', description: 'Resume an interrupted training job.' },
  resumeCancelled: { type: 'boolean', description: 'Resume a cancelled training job.' },
  auditEveryStep: { type: 'integer', description: 'Audit-log cadence in training steps.' },
  gates: { type: 'object', description: 'Predeclared evaluation gate thresholds.' },
  reportFile: { type: 'string', description: 'Evaluation report path.' },
  maxNewTokens: { type: 'integer', description: 'Generation budget per eval sample.' },
  background: { type: 'boolean', description: 'Run in an independent worker and return a request id.' },
  retry: { type: 'boolean', description: 'Explicitly retry a failed or interrupted request.' },
  requestId: { type: 'string', description: 'Background request id.' },
  evalReport: { type: 'string', description: 'Evaluation report consumed by export.' },
  qualityReport: { type: 'string', description: 'External quality report consumed by export.' },
  version: { type: 'string', description: 'Export version label.' },
  evaluateIfMissing: { type: 'boolean', description: 'Run evaluation when no report exists.' },
});

export function parametersFor(action) {
  const spec = Object.hasOwn(ACTION_SPECS, action) ? ACTION_SPECS[action] : null;
  if (!spec) throw new Error(`unknown action "${action}"`);
  const properties = {};
  for (const param of spec.params) {
    const schema = PARAM_SCHEMA[param];
    if (!schema) throw new Error(`no JSON schema declared for param "${param}" of action "${action}"`);
    properties[param] = schema;
  }
  return {
    type: 'object', additionalProperties: false, properties,
    ...(spec.required?.length ? { required: [...spec.required] } : {}),
  };
}

function fail(action, code, message) {
  return { ok: false, contractVersion: CONTRACT_VERSION, action, error: { code, message } };
}

/**
 * Build a harness-neutral operation client bound to one workspace.
 *
 * `allowCompute: false` lets a host mount the client for inspection only —
 * doctor and status still answer, while anything that would burn compute or
 * write artifacts is refused with `compute-not-authorized` instead of running.
 */
export function createOperationClient({ baseDir = process.cwd(), configFile = null, allowCompute = true } = {}) {
  const ctx = { baseDir, configFile };

  async function invoke(action, params = {}) {
    if (typeof action !== 'string' || !action) {
      return fail(String(action), 'invalid-action', 'action must be a non-empty string');
    }
    const spec = Object.hasOwn(ACTION_SPECS, action) ? ACTION_SPECS[action] : null;
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
      return fail(action, 'compute-not-authorized', `action "${action}" consumes compute and this client was mounted read-only`);
    }
    try {
      const result = await spec.handler(ctx, params);
      return { ok: true, contractVersion: CONTRACT_VERSION, action, result };
    } catch (err) {
      const code = err?.code && typeof err.code === 'string' ? err.code : err?.name === 'ConfigError' ? 'config-error' : 'action-failed';
      return fail(action, code, err?.message ?? String(err));
    }
  }

  return { baseDir, allowCompute, actions: ACTION_NAMES, invoke };
}
