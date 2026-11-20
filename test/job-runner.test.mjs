import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {
  runJob, readJobState, readAuditLog, assertTransition, isTerminal, redactForAudit,
  JobBudget, BudgetExceededError, JobCancelledError, JOB_STATES,
} from '../src/jobs/runner.mjs';

const ROOT = '.tmf-run/test-job-runner';

async function freshRoot(name) {
  const root = `${ROOT}/${name}`;
  await fs.rm(root, { recursive: true, force: true });
  return root;
}

function counterSteps(log) {
  return [
    { name: 'prepare', run: async () => { log.push('prepare'); return { records: 2 }; } },
    { name: 'train', run: async ({ outputs }) => { log.push('train'); return { steps: outputs.prepare.records * 2 }; } },
    { name: 'report', run: async ({ outputs }) => { log.push('report'); return { trained: outputs.train.steps }; } },
  ];
}

test('state machine only allows declared transitions', () => {
  assert.equal(assertTransition('pending', 'running'), 'running');
  assert.equal(assertTransition('running', 'budget_exceeded'), 'budget_exceeded');
  assert.throws(() => assertTransition('succeeded', 'running'), /illegal job state transition/);
  assert.throws(() => assertTransition('pending', 'succeeded'), /illegal job state transition/);
  assert.throws(() => assertTransition('running', 'paused'), /unknown job state/);
  assert.equal(isTerminal('running'), false);
  assert.ok(JOB_STATES.every((s) => typeof s === 'string'));
});

test('a job runs every step, persists state and writes an audit trail', async () => {
  const root = await freshRoot('happy');
  const log = [];
  const result = await runJob({ jobId: 'job-1', kind: 'train', root, steps: counterSteps(log), input: { seed: 7 } });

  assert.equal(result.state, 'succeeded');
  assert.equal(result.replayed, false);
  assert.deepEqual(result.result, { trained: 4 });
  assert.deepEqual(log, ['prepare', 'train', 'report']);

  const state = await readJobState(root, 'job-1');
  assert.equal(state.state, 'succeeded');
  assert.deepEqual(state.completedSteps.map((s) => s.name), ['prepare', 'train', 'report']);

  const audit = await readAuditLog(root, 'job-1');
  const events = audit.map((row) => row.event);
  assert.equal(events[0], 'job.started');
  assert.equal(events.at(-1), 'job.succeeded');
  assert.equal(events.filter((e) => e === 'step.succeeded').length, 3);
  assert.ok(audit.every((row) => typeof row.ts === 'string' && row.jobId === 'job-1'));
});

test('replaying a succeeded job reuses the stored result without redoing side effects', async () => {
  const root = await freshRoot('replay');
  const log = [];
  await runJob({ jobId: 'job-1', kind: 'train', root, steps: counterSteps(log), input: { seed: 7 } });
  const again = await runJob({ jobId: 'job-1', kind: 'train', root, steps: counterSteps(log), input: { seed: 7 } });

  assert.equal(again.replayed, true);
  assert.equal(again.state, 'succeeded');
  assert.deepEqual(again.result, { trained: 4 });
  assert.deepEqual(log, ['prepare', 'train', 'report'], 'no step re-executed on replay');
});

test('reusing a job id with different input is rejected instead of corrupting history', async () => {
  const root = await freshRoot('digest');
  const log = [];
  await runJob({ jobId: 'job-1', kind: 'train', root, steps: counterSteps(log), input: { seed: 7 } });
  await assert.rejects(
    runJob({ jobId: 'job-1', kind: 'train', root, steps: counterSteps(log), input: { seed: 8 } }),
    /different input\/plan digest/,
  );
});

test('a crashed job resumes from its last completed step', async () => {
  const root = await freshRoot('resume');
  const log = [];
  const failing = [
    { name: 'prepare', run: async () => { log.push('prepare'); return { records: 2 }; } },
    { name: 'train', run: async () => { log.push('train-boom'); throw new Error('disk full'); } },
    { name: 'report', run: async () => { log.push('report'); return { ok: true }; } },
  ];
  await assert.rejects(runJob({ jobId: 'job-2', kind: 'train', root, steps: failing, input: {} }), /disk full/);

  let failed = await readJobState(root, 'job-2');
  assert.equal(failed.state, 'failed');
  assert.equal(failed.error.step, 'train');
  assert.deepEqual(failed.completedSteps.map((s) => s.name), ['prepare']);

  const fixed = [
    failing[0],
    { name: 'train', run: async ({ outputs }) => { log.push('train-ok'); return { steps: outputs.prepare.records }; } },
    { name: 'report', run: async ({ outputs }) => { log.push('report'); return { trained: outputs.train.steps }; } },
  ];
  // The failed state is terminal, so resuming is a fresh attempt over the same
  // job directory: completed steps are skipped, the rest runs.
  await fs.writeFile(`${root}/job-2/state.json`, JSON.stringify({ ...failed, state: 'pending' }));
  const resumed = await runJob({ jobId: 'job-2', kind: 'train', root, steps: fixed, input: {} });

  assert.equal(resumed.state, 'succeeded');
  assert.equal(resumed.attempt, 2);
  assert.deepEqual(resumed.result, { trained: 2 });
  assert.deepEqual(log, ['prepare', 'train-boom', 'train-ok', 'report'], 'prepare was not re-run');
  const audit = await readAuditLog(root, 'job-2');
  assert.ok(audit.some((row) => row.event === 'job.resumed'));
  assert.ok(audit.some((row) => row.event === 'step.skipped' && row.step === 'prepare'));
});

test('cancellation stops the job at a step boundary and records it', async () => {
  const root = await freshRoot('cancel');
  const controller = new AbortController();
  const log = [];
  const steps = [
    { name: 'prepare', run: async () => { log.push('prepare'); controller.abort('operator cancelled'); return { records: 1 }; } },
    { name: 'train', run: async () => { log.push('train'); return { steps: 1 }; } },
  ];
  await assert.rejects(
    runJob({ jobId: 'job-3', kind: 'train', root, steps, signal: controller.signal }),
    (err) => err instanceof JobCancelledError && err.code === 'JOB_CANCELLED',
  );
  assert.deepEqual(log, ['prepare'], 'no step started after cancellation');
  const state = await readJobState(root, 'job-3');
  assert.equal(state.state, 'cancelled');
  assert.deepEqual(state.completedSteps.map((s) => s.name), ['prepare']);
  const audit = await readAuditLog(root, 'job-3');
  assert.ok(audit.some((row) => row.event === 'job.cancelled'));
});

test('budget caps stop a job before the over-budget step spends anything', async () => {
  const root = await freshRoot('budget');
  const log = [];
  const steps = [
    { name: 'a', cost: { tokens: 400 }, run: async () => { log.push('a'); return 1; } },
    { name: 'b', cost: { tokens: 400 }, run: async () => { log.push('b'); return 2; } },
    { name: 'c', cost: { tokens: 400 }, run: async () => { log.push('c'); return 3; } },
  ];
  await assert.rejects(
    runJob({ jobId: 'job-4', kind: 'train', root, steps, budget: { maxTokens: 900 } }),
    (err) => err instanceof BudgetExceededError && err.details.limit === 'maxTokens',
  );
  assert.deepEqual(log, ['a', 'b'], 'the step that would exceed the cap never ran');
  const state = await readJobState(root, 'job-4');
  assert.equal(state.state, 'budget_exceeded');
  assert.equal(state.budget.usage.tokens, 800);
  const audit = await readAuditLog(root, 'job-4');
  assert.ok(audit.some((row) => row.event === 'job.budget_exceeded' && row.limit === 'maxTokens'));
});

test('step count and cost caps are enforced independently', () => {
  const budget = new JobBudget({ maxSteps: 2, maxCostUsd: 1 }).start(0);
  budget.charge({ steps: 1, costUsd: 0.4 }, 0);
  budget.charge({ steps: 1, costUsd: 0.4 }, 0);
  assert.throws(() => budget.charge({ steps: 1 }, 0), /steps 3 > 2/);
  const wall = new JobBudget({ maxWallClockMs: 50 }).start(0);
  assert.throws(() => wall.charge({ steps: 1 }, 100), /wallClockMs 100 > 50/);
});

test('audit payloads are redacted: credential keys dropped, PII masked', async () => {
  const redacted = redactForAudit({
    apiKey: 'sk-live-123456',
    nested: { authorization: 'Bearer abc', note: 'contact alice@example.com now' },
    list: ['token ok', 'ring 4111 1111 1111 1111'],
    count: 3,
  });
  assert.equal(redacted.apiKey, '[REDACTED]');
  assert.equal(redacted.nested.authorization, '[REDACTED]');
  assert.ok(!redacted.nested.note.includes('alice@example.com'));
  assert.ok(!redacted.list[1].includes('4111 1111 1111 1111'));
  assert.equal(redacted.count, 3);

  const root = await freshRoot('redact');
  const steps = [{
    name: 'leaky',
    run: async ({ audit }) => {
      await audit('step.note', { password: 'hunter2', email: 'bob@example.com' });
      return { ok: true };
    },
  }];
  await runJob({ jobId: 'job-5', kind: 'train', root, steps });
  const raw = await fs.readFile(`${root}/job-5/audit.jsonl`, 'utf8');
  assert.ok(!raw.includes('hunter2'));
  assert.ok(!raw.includes('bob@example.com'));
  assert.ok(raw.includes('[REDACTED]'));
});

test('runJob validates its own contract', async () => {
  const root = await freshRoot('validate');
  await assert.rejects(runJob({ kind: 'train', root, steps: [{ name: 'a', run: async () => 1 }] }), /requires jobId/);
  await assert.rejects(runJob({ jobId: 'j', kind: 'train', root, steps: [] }), /at least one step/);
  await assert.rejects(runJob({ jobId: 'j', kind: 'train', root, steps: [{ name: 'a' }] }), /requires \{ name, run \}/);
});
