// Auditable job orchestration.
//
// Every long-running foundry job (data prep, training, evaluation, export) runs
// through this runner so the operator gets the same guarantees regardless of
// which job it is: a durable state machine, an append-only audit log that is
// safe to hand to a reviewer, resume after a crash, cooperative cancellation,
// hard budget caps, and replay that does not redo completed side effects.

import path from 'node:path';
import { appendJsonl, ensureDir, readJsonIfExists, readJsonl, writeJson } from '../core/fsx.mjs';
import { sha256Json } from '../core/hash.mjs';
import { redactPii } from '../data/pii.mjs';

export const JOB_STATES = ['pending', 'running', 'succeeded', 'failed', 'cancelled', 'budget_exceeded'];

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'cancelled', 'budget_exceeded']);

// Only these transitions are legal. An illegal transition is a bug in a job
// implementation, so it throws rather than being recorded as job failure.
const TRANSITIONS = {
  pending: ['running', 'cancelled'],
  running: ['running', 'succeeded', 'failed', 'cancelled', 'budget_exceeded'],
  succeeded: [],
  failed: [],
  // Operator cancellation is a pause, not a verdict: the completed steps are
  // still valid, so an explicit `resumeCancelled` run continues the same job.
  cancelled: ['running'],
  budget_exceeded: [],
};

// Matched against the normalized key, exactly: a budget field named `tokens`
// or `maxTokens` must stay readable while an `accessToken` must not.
const SECRET_KEYS = new Set([
  'token', 'accesstoken', 'refreshtoken', 'idtoken', 'authtoken', 'sessiontoken',
  'secret', 'clientsecret', 'password', 'passwd', 'pwd', 'passphrase',
  'apikey', 'apisecret', 'credential', 'credentials', 'authorization', 'auth',
  'cookie', 'bearer', 'privatekey', 'sessionkey', 'signingkey',
]);
const REDACTED = '[REDACTED]';

export class JobCancelledError extends Error {
  constructor(message = 'job cancelled') {
    super(message);
    this.name = 'JobCancelledError';
    this.code = 'JOB_CANCELLED';
  }
}

export class BudgetExceededError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'BudgetExceededError';
    this.code = 'JOB_BUDGET_EXCEEDED';
    this.details = details;
  }
}

export function assertTransition(from, to) {
  if (!JOB_STATES.includes(to)) throw new Error(`unknown job state: ${to}`);
  const allowed = TRANSITIONS[from] ?? [];
  if (!allowed.includes(to)) {
    throw new Error(`illegal job state transition: ${from} -> ${to}`);
  }
  return to;
}

export function isTerminal(state) {
  return TERMINAL_STATES.has(state);
}

// Redaction is applied to everything that reaches the audit log. Keys that name
// a credential are dropped wholesale (the value shape itself can leak); free
// text goes through the same PII rules the data pipeline uses.
export function isSecretKey(key) {
  return SECRET_KEYS.has(String(key).replace(/[^a-z0-9]/gi, '').toLowerCase());
}

export function redactForAudit(value, depth = 0) {
  if (depth > 8) return '[TRUNCATED]';
  if (value === null || value === undefined) return value ?? null;
  if (typeof value === 'string') return redactPii(value).text;
  if (typeof value === 'number' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map((item) => redactForAudit(item, depth + 1));
  if (typeof value === 'object') {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      out[key] = isSecretKey(key) ? REDACTED : redactForAudit(item, depth + 1);
    }
    return out;
  }
  return String(value);
}

export class JobBudget {
  constructor({ maxSteps = null, maxWallClockMs = null, maxTokens = null, maxCostUsd = null } = {}) {
    this.limits = { maxSteps, maxWallClockMs, maxTokens, maxCostUsd };
    this.usage = { steps: 0, wallClockMs: 0, tokens: 0, costUsd: 0 };
    this.startedAt = null;
  }

  start(now) {
    this.startedAt = now;
    return this;
  }

  // Charged before a step runs so an over-budget job stops instead of spending.
  charge({ steps = 0, tokens = 0, costUsd = 0 } = {}, now = Date.now()) {
    const next = {
      steps: this.usage.steps + steps,
      tokens: this.usage.tokens + tokens,
      costUsd: this.usage.costUsd + costUsd,
      wallClockMs: this.startedAt === null ? 0 : now - this.startedAt,
    };
    const checks = [
      ['maxSteps', next.steps, 'steps'],
      ['maxTokens', next.tokens, 'tokens'],
      ['maxCostUsd', next.costUsd, 'costUsd'],
      ['maxWallClockMs', next.wallClockMs, 'wallClockMs'],
    ];
    for (const [limitKey, value, usageKey] of checks) {
      const limit = this.limits[limitKey];
      if (limit !== null && limit !== undefined && value > limit) {
        throw new BudgetExceededError(`budget exceeded: ${usageKey} ${value} > ${limit}`, {
          limit: limitKey,
          limitValue: limit,
          attempted: value,
          usage: { ...this.usage },
        });
      }
    }
    this.usage = next;
    return this.usage;
  }

  snapshot(now = Date.now()) {
    return {
      limits: { ...this.limits },
      usage: { ...this.usage, wallClockMs: this.startedAt === null ? this.usage.wallClockMs : now - this.startedAt },
    };
  }
}

function jobPaths(root, jobId) {
  const dir = path.join(root, jobId);
  return {
    dir,
    state: path.join(dir, 'state.json'),
    audit: path.join(dir, 'audit.jsonl'),
    result: path.join(dir, 'result.json'),
  };
}

export async function readJobState(root, jobId) {
  return readJsonIfExists(jobPaths(root, jobId).state, null);
}

export async function readAuditLog(root, jobId) {
  const rows = await readJsonl(jobPaths(root, jobId).audit).catch(() => []);
  return rows;
}

// A job is a declarative list of named steps. Each step's recorded output is the
// resume unit: on replay a step whose record already exists is skipped and its
// stored output is reused, so completed side effects are never repeated.
export async function runJob({
  jobId,
  kind,
  root,
  steps,
  input = {},
  budget = {},
  signal = null,
  now = () => Date.now(),
  resume = true,
  resumeCancelled = false,
} = {}) {
  if (!jobId) throw new Error('runJob requires jobId');
  if (!kind) throw new Error('runJob requires kind');
  if (!root) throw new Error('runJob requires root');
  if (!Array.isArray(steps) || steps.length === 0) throw new Error('runJob requires at least one step');

  const paths = jobPaths(root, jobId);
  await ensureDir(paths.dir);

  const inputDigest = sha256Json(input);
  const planDigest = sha256Json(steps.map((step) => ({ name: step.name, cost: step.cost ?? null })));

  const prior = resume ? await readJsonIfExists(paths.state, null) : null;
  if (prior) {
    if (prior.inputDigest !== inputDigest || prior.planDigest !== planDigest) {
      throw new Error(
        `job ${jobId} already exists with a different input/plan digest; use a new jobId instead of mutating history`,
      );
    }
    // Idempotency: a finished job is returned as-is, not re-executed. The one
    // exception is a cancelled job the caller explicitly asks to continue,
    // which picks up at the first step that never completed.
    const continuable = prior.state === 'cancelled' && resumeCancelled;
    if (isTerminal(prior.state) && !continuable) {
      const result = await readJsonIfExists(paths.result, null);
      return { ...prior, replayed: true, result: result?.result ?? null };
    }
  }

  const jobBudget = new JobBudget({ ...(prior?.budget?.limits ?? {}), ...budget });
  jobBudget.usage = { ...jobBudget.usage, ...(prior?.budget?.usage ?? {}) };
  const startedAt = prior?.startedAt ?? new Date(now()).toISOString();
  jobBudget.start(now());

  let state = {
    version: 1,
    jobId,
    kind,
    state: prior?.state ?? 'pending',
    attempt: (prior?.attempt ?? 0) + 1,
    inputDigest,
    planDigest,
    startedAt,
    updatedAt: startedAt,
    completedSteps: prior?.completedSteps ?? [],
    budget: jobBudget.snapshot(now()),
    error: null,
  };

  const audit = async (event, payload = {}) => {
    await appendJsonl(paths.audit, {
      ts: new Date(now()).toISOString(),
      jobId,
      kind,
      attempt: state.attempt,
      event,
      ...redactForAudit(payload),
    });
  };

  const persist = async (nextState, extra = {}) => {
    if (nextState && nextState !== state.state) assertTransition(state.state, nextState);
    state = {
      ...state,
      ...extra,
      state: nextState ?? state.state,
      budget: jobBudget.snapshot(now()),
      updatedAt: new Date(now()).toISOString(),
    };
    await writeJson(paths.state, state);
    return state;
  };

  const checkCancel = async () => {
    if (signal?.aborted) {
      await audit('job.cancelled', { reason: signal.reason ? String(signal.reason) : 'signal aborted' });
      await persist('cancelled', { error: { code: 'JOB_CANCELLED', message: 'job cancelled' } });
      throw new JobCancelledError();
    }
  };

  await audit(prior ? 'job.resumed' : 'job.started', {
    inputDigest,
    planDigest,
    completedSteps: state.completedSteps.map((s) => s.name),
    budgetLimits: jobBudget.limits,
  });
  await checkCancel();
  await persist('running');

  const outputs = {};
  for (const done of state.completedSteps) outputs[done.name] = done.output ?? null;

  for (const step of steps) {
    if (!step?.name || typeof step.run !== 'function') {
      throw new Error('each step requires { name, run }');
    }
    const already = state.completedSteps.find((s) => s.name === step.name);
    if (already) {
      await audit('step.skipped', { step: step.name, reason: 'already completed in a prior attempt' });
      continue;
    }
    await checkCancel();
    try {
      jobBudget.charge({ steps: 1, ...(step.cost ?? {}) }, now());
    } catch (error) {
      if (error instanceof BudgetExceededError) {
        await audit('job.budget_exceeded', { step: step.name, ...error.details });
        await persist('budget_exceeded', { error: { code: error.code, message: error.message, details: error.details } });
        throw error;
      }
      throw error;
    }
    await audit('step.started', { step: step.name });
    let output;
    try {
      output = await step.run({ input, outputs, signal, audit, budget: jobBudget });
    } catch (error) {
      if (error instanceof JobCancelledError || signal?.aborted) {
        await audit('job.cancelled', { step: step.name, reason: error.message });
        await persist('cancelled', { error: { code: 'JOB_CANCELLED', message: error.message } });
        throw error instanceof JobCancelledError ? error : new JobCancelledError(error.message);
      }
      if (error instanceof BudgetExceededError) {
        await audit('job.budget_exceeded', { step: step.name, ...error.details });
        await persist('budget_exceeded', { error: { code: error.code, message: error.message, details: error.details } });
        throw error;
      }
      await audit('step.failed', { step: step.name, error: error.message });
      await persist('failed', { error: { code: error.code ?? 'STEP_FAILED', message: error.message, step: step.name } });
      throw error;
    }
    outputs[step.name] = output ?? null;
    state.completedSteps = [
      ...state.completedSteps,
      { name: step.name, completedAt: new Date(now()).toISOString(), output: output ?? null },
    ];
    await audit('step.succeeded', { step: step.name, outputDigest: sha256Json(output ?? null) });
    await persist(null, { completedSteps: state.completedSteps });
  }

  await checkCancel();
  const result = outputs[steps[steps.length - 1].name] ?? null;
  await writeJson(paths.result, { jobId, kind, result, resultDigest: sha256Json(result) });
  await audit('job.succeeded', { resultDigest: sha256Json(result) });
  await persist('succeeded');
  return { ...state, replayed: false, result };
}
