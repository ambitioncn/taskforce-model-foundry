import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { synthesize } from '../src/api.mjs';
import { loadCampaign, runCampaign, campaignStatus } from '../src/controller/campaign.mjs';

async function fixture(t, { candidateCount = 2, budgetSteps = 8 } = {}) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-campaign-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  await synthesize({ baseDir: dir, taskCount: 100, pretrainCount: 100, includeNoise: false });
  const candidates = [];
  for (let i = 0; i < candidateCount; i++) {
    const configFile = `candidate-${i}.json`;
    await fs.writeFile(path.join(dir, configFile), JSON.stringify({
      format: 'tmf-config-v1', project: 'campaign-fixture', runRoot: '.tmf-run',
      backend: 'builtin-js', mode: 'lora', seed: `candidate-${i}`,
      model: { maxVocab: 256, maxSeqLen: 96 },
      data: { datasets: [
        { name: 'task', path: 'data/tickets.jsonl', format: 'jsonl', license: 'synthetic' },
        { name: 'pretrain', path: 'data/corpus.jsonl', format: 'jsonl', license: 'synthetic' },
      ], config: {} },
      hyper: { steps: 2, lr: 0.005, batchSize: 4 },
      budget: { maxTokens: 200000, maxWallClockMs: 120000 },
      eval: { gates: [{ id: 'loss-cap', metric: 'tuned.holdout.loss', comparator: 'lte', threshold: 100 }] },
    }));
    candidates.push({ id: `c${i}`, configFile });
  }
  const planFile = path.join(dir, 'campaign.json');
  await fs.writeFile(planFile, JSON.stringify({ format: 'tmf-campaign-v1', id: 'fixture', candidates,
    budget: { maxTrials: candidateCount, maxTotalTrainSteps: budgetSteps, deadlineAt: '2099-01-01T00:00:00Z' },
    policy: { stopOnTechnicalPass: true, rankMetric: 'tuned.holdout.loss', direction: 'min' },
  }));
  return { dir, planFile };
}

test('campaign autonomously trains, evaluates and stops after technical success', async (t) => {
  const { planFile } = await fixture(t);
  const first = await runCampaign({ planFile });
  assert.equal(first.state, 'succeeded');
  assert.equal(first.result.status, 'technical_passed');
  assert.equal(first.result.selected.id, 'c0');
  assert.equal(first.result.trials[1].state, 'skipped_goal_reached');
  assert.equal(first.result.exported, false);
  const replay = await runCampaign({ planFile });
  assert.equal(replay.replayed, true);
  assert.deepEqual(replay.result, first.result);
  assert.equal((await campaignStatus({ planFile })).state.state, 'succeeded');
});

test('campaign rejects post-hoc gate or data edits and freezes the budget', async (t) => {
  const { dir, planFile } = await fixture(t, { budgetSteps: 1 });
  await assert.rejects(() => loadCampaign(planFile), /cannot fund the first candidate/);
  const plan = JSON.parse(await fs.readFile(planFile));
  plan.budget.maxTotalTrainSteps = 8;
  await fs.writeFile(planFile, JSON.stringify(plan));
  await runCampaign({ planFile });
  const configFile = path.join(dir, 'candidate-0.json');
  const config = JSON.parse(await fs.readFile(configFile));
  config.eval.gates[0].threshold = 101;
  await fs.writeFile(configFile, JSON.stringify(config));
  await assert.rejects(() => runCampaign({ planFile }), /different input\/plan digest|candidates must share/);
});

test('standalone background campaign deduplicates and reaches a durable decision', async (t) => {
  const { planFile } = await fixture(t);
  const old = process.env.TMF_BACKGROUND_LAUNCHER;
  process.env.TMF_BACKGROUND_LAUNCHER = 'direct';
  t.after(() => { if (old === undefined) delete process.env.TMF_BACKGROUND_LAUNCHER; else process.env.TMF_BACKGROUND_LAUNCHER = old; });
  const { startCampaign, backgroundCampaignStatus } = await import('../src/controller/background.mjs');
  const first = await startCampaign({ planFile });
  const second = await startCampaign({ planFile });
  assert.equal(first.requestId, second.requestId);
  assert.equal(second.reused, true);
  let final;
  // Multiple background fixtures share constrained CI runners; assert a
  // terminal receipt within a bounded wall-clock window, not 15 seconds of
  // scheduler-dependent polling.
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    final = await backgroundCampaignStatus({ planFile });
    if (['succeeded', 'failed', 'interrupted'].includes(final.state)) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  assert.equal(final.state, 'succeeded', JSON.stringify({ state: final.run, worker: final.workerResult }));
  assert.equal(final.decision.status, 'technical_passed');
  assert.equal(final.workerResult.state, 'succeeded');
  const replay = await startCampaign({ planFile });
  assert.equal(replay.reused, true);
  assert.equal(replay.state, 'succeeded');
});

test('CLI runs and queries a campaign without an agent harness', async (t) => {
  const { dir, planFile } = await fixture(t, { candidateCount: 1 });
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const exec = promisify(execFile);
  const cli = path.resolve('bin/tmf.mjs');
  const run = await exec(process.execPath, [cli, 'campaign', 'run', '--plan', planFile, '--json'], { cwd: dir, timeout: 30000 });
  assert.equal(JSON.parse(run.stdout).result.status, 'technical_passed');
  const status = await exec(process.execPath, [cli, 'campaign', 'status', '--plan', planFile, '--json'], { cwd: dir, timeout: 30000 });
  assert.equal(JSON.parse(status.stdout).decision.status, 'technical_passed');
});

test('campaign bounds work when no candidate passes and never exports', async (t) => {
  const { dir, planFile } = await fixture(t, { candidateCount: 1 });
  const configFile = path.join(dir, 'candidate-0.json');
  const config = JSON.parse(await fs.readFile(configFile));
  config.eval.gates[0].threshold = -1;
  await fs.writeFile(configFile, JSON.stringify(config));
  const result = await runCampaign({ planFile });
  assert.equal(result.state, 'succeeded');
  assert.equal(result.result.status, 'no_candidate_passed');
  assert.equal(result.result.selected, null);
  assert.equal(result.result.exported, false);
  assert.equal(result.result.stepsCharged, 2);
});

test('cancelled campaign requires explicit resume and keeps frozen inputs', async (t) => {
  const { planFile } = await fixture(t, { candidateCount: 1 });
  const controller = new AbortController();
  controller.abort(new Error('operator cancelled'));
  await assert.rejects(() => runCampaign({ planFile, signal: controller.signal }), { code: 'JOB_CANCELLED' });
  const cancelled = await campaignStatus({ planFile });
  assert.equal(cancelled.state.state, 'cancelled');
  const unchanged = await runCampaign({ planFile });
  assert.equal(unchanged.state, 'cancelled');
  assert.equal(unchanged.replayed, true);
  const resumed = await runCampaign({ planFile, resumeCancelled: true });
  assert.equal(resumed.state, 'succeeded');
  assert.equal(resumed.result.status, 'technical_passed');
});
