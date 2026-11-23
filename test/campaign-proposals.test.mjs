import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { synthesize } from '../src/api.mjs';
import { buildProposalPool, chooseNextProposal, chooseReflectiveProposal, reflectOnFailedTrial, gateFailureSeverity } from '../src/controller/index.mjs';
import { loadCampaign, runCampaign } from '../src/controller/campaign.mjs';

const baseRaw = {
  format: 'tmf-config-v1', project: 'proposal-fixture', runRoot: '.tmf-run',
  backend: 'builtin-js', mode: 'lora', seed: 'proposal-fixture',
  model: { maxVocab: 256, maxSeqLen: 96 },
  data: { datasets: [
    { name: 'task', path: 'data/tickets.jsonl', format: 'jsonl', license: 'synthetic' },
    { name: 'pretrain', path: 'data/corpus.jsonl', format: 'jsonl', license: 'synthetic' },
  ], config: {} },
  hyper: { steps: 2, lr: 0.005, batchSize: 4 },
  budget: { maxTokens: 200000, maxWallClockMs: 120000 },
  eval: { gates: [{ id: 'loss-cap', metric: 'tuned.holdout.loss', comparator: 'lte', threshold: 100 }] },
};

const space = { baseConfigFile: 'base.json', strategy: 'best-neighbor', parameters: [
  { path: 'hyper.lr', values: [0.001, 0.005, 0.01] },
  { path: 'hyper.batchSize', values: [2, 4, 8] },
] };

test('proposal selection follows the best measured candidate, never repeats a candidate', () => {
  const built = buildProposalPool({ proposal: space, baseRaw, baseDir: '/tmp' });
  const first = chooseNextProposal({ ...built, previous: [], direction: 'min' });
  assert.equal(first.id, 'p-1-1');
  const second = chooseNextProposal({ ...built, previous: [{ id: first.id, score: 1 }], direction: 'min' });
  assert.equal(second.id, 'p-0-1');
  const third = chooseNextProposal({ ...built, previous: [
    { id: first.id, score: 1 }, { id: second.id, score: 0.5 },
  ], direction: 'min' });
  assert.equal(third.id, 'p-0-0');
  assert.equal(third.parentId, second.id);
  assert.equal(new Set([first.id, second.id, third.id]).size, 3);
});

test('proposal refuses edits outside approved numeric hyperparameters', () => {
  for (const forbidden of ['eval.gates', 'data.datasets', 'model.maxSeqLen', 'hyper.steps']) {
    assert.throws(() => buildProposalPool({ proposal: { ...space, parameters: [{ path: forbidden, values: [1, 2] }] },
      baseRaw, baseDir: '/tmp' }), /not allowed/);
  }
  assert.throws(() => buildProposalPool({ proposal: { ...space, parameters: [{ path: 'hyper.lr', values: [0.005, 0.001] }] },
    baseRaw, baseDir: '/tmp' }), /strictly increasing/);
});

test('reflective search records failed evidence and jumps beyond neighbors', () => {
  const built = buildProposalPool({ proposal: { ...space, strategy: 'reflective' }, baseRaw, baseDir: '/tmp' });
  const first = chooseReflectiveProposal({ ...built, previous: [] });
  assert.equal(first.id, 'p-1-1');
  const report = { reportDigest: 'frozen-report', gates: { results: [
    { id: 'loss', metric: 'tuned.holdout.loss', actual: 2, comparator: 'lte', threshold: 1, ok: false, reason: 'loss too high' },
  ] } };
  const reflection = reflectOnFailedTrial({ pool: built.pool, previous: [], current: { ...first, score: 2 }, report, direction: 'min' });
  assert.equal(reflection.observation.failedGates[0].threshold, 1);
  assert.equal(reflection.action.method, 'contrast');
  const second = chooseReflectiveProposal({ ...built, previous: [{ ...first, score: 2, reflection }] });
  assert.equal(second.id, 'p-0-0'); // two dimensions away, not a neighbor
  assert.equal(reflection.action.nextCandidateId, second.id);
  assert.deepEqual(Object.keys(reflection.action.changes), ['hyper.lr', 'hyper.batchSize']);
  const failedAgain = reflectOnFailedTrial({ pool: built.pool, previous: [{ ...first, score: 2, reflection }],
    current: { ...second, score: 1.5 }, report, direction: 'min' });
  assert.equal(failedAgain.observation.bestPrior.id, first.id);
  assert.equal(failedAgain.action.method, 'pivot'); // improved but no farther point in that direction
});

test('reflective search stops rather than pretending hyperparameters fix missing evidence', () => {
  const built = buildProposalPool({ proposal: { ...space, strategy: 'reflective' }, baseRaw, baseDir: '/tmp' });
  const first = chooseReflectiveProposal({ ...built, previous: [] });
  const reflection = reflectOnFailedTrial({ pool: built.pool, previous: [], current: { ...first, score: 1 },
    report: { reportDigest: 'missing-samples', gates: { results: [
      { id: 'exact-match', metric: 'tuned.holdout.exactMatch.accuracy', actual: 0.9,
        comparator: 'gte', threshold: 0.25, samples: 2, minSamples: 10, ok: false },
    ] } }, direction: 'max' });
  assert.equal(reflection.action.method, 'stop_unactionable');
  assert.equal(chooseReflectiveProposal({ ...built, previous: [{ ...first, score: 1, reflection }] }), null);
  assert.equal(chooseReflectiveProposal({ ...built, previous: [
    { ...first, score: 1, reflection }, { state: 'skipped_unactionable_evidence' },
  ] }), null);
});

test('reflection prioritizes frozen gate failures over a misleading rank metric', () => {
  assert.equal(gateFailureSeverity([{ actual: 0.2, comparator: 'gte', threshold: 0.5, ok: false }]), 0.3);
  const built = buildProposalPool({ proposal: { ...space, strategy: 'reflective' }, baseRaw, baseDir: '/tmp' });
  const prior = built.pool.find((item) => item.id === 'p-0-1');
  const current = built.pool.find((item) => item.id === 'p-1-1');
  const result = reflectOnFailedTrial({ pool: built.pool,
    previous: [{ ...prior, score: 0.1, gateSeverity: 0.6 }],
    current: { ...current, score: 0.2, gateSeverity: 0.3 },
    report: { reportDigest: 'gate-evidence', gates: { results: [
      { id: 'quality', metric: 'tuned.quality', actual: 0.2, comparator: 'gte', threshold: 0.5, ok: false },
    ] } }, direction: 'min' });
  assert.equal(result.action.method, 'extrapolate');
  assert.equal(result.action.nextCandidateId, 'p-2-0');
  assert.equal(result.observation.gateSeverity, 0.3);
});

async function fixture(t) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-proposal-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  await synthesize({ baseDir: dir, taskCount: 100, pretrainCount: 100, includeNoise: false });
  await fs.writeFile(path.join(dir, 'base.json'), JSON.stringify(baseRaw));
  const planFile = path.join(dir, 'campaign.json');
  await fs.writeFile(planFile, JSON.stringify({ format: 'tmf-campaign-v1', id: 'proposal-run',
    proposal: { baseConfigFile: 'base.json', strategy: 'best-neighbor', parameters: [
      { path: 'hyper.lr', values: [0.001, 0.005, 0.01] },
    ] },
    budget: { maxTrials: 2, maxTotalTrainSteps: 4, deadlineAt: '2099-01-01T00:00:00Z' },
    policy: { stopOnTechnicalPass: false, rankMetric: 'tuned.holdout.loss', direction: 'min' },
  }));
  return { dir, planFile };
}

test('campaign creates reviewable configs and evaluates new candidates within budget', async (t) => {
  const { dir, planFile } = await fixture(t);
  const run = await runCampaign({ planFile });
  assert.equal(run.state, 'succeeded');
  assert.deepEqual(run.result.trials.map((trial) => trial.id), ['p-1', 'p-0']);
  assert.equal(run.result.stepsCharged, 4);
  assert.equal(run.result.exported, false);
  const first = JSON.parse(await fs.readFile(run.result.trials[0].configFile));
  const second = JSON.parse(await fs.readFile(run.result.trials[1].configFile));
  assert.equal(first.hyper.lr, 0.005);
  assert.equal(second.hyper.lr, 0.001);
  assert.equal(second.eval.gates[0].threshold, first.eval.gates[0].threshold);
  assert.deepEqual(second.data, first.data);
  assert.equal(run.result.trials[1].proposal.parentId, 'p-1');
  assert.equal((await runCampaign({ planFile })).replayed, true);
  const generatedFile = run.result.trials[0].configFile;
  const generated = JSON.parse(await fs.readFile(generatedFile));
  generated.hyper.lr = 0.01;
  await fs.writeFile(generatedFile, JSON.stringify(generated));
  await assert.rejects(() => runCampaign({ planFile }), /generated candidate file was changed/);
  await fs.writeFile(generatedFile, JSON.stringify(first));
  const raw = JSON.parse(await fs.readFile(path.join(dir, 'base.json')));
  raw.eval.gates[0].threshold = 101;
  await fs.writeFile(path.join(dir, 'base.json'), JSON.stringify(raw));
  await assert.rejects(() => runCampaign({ planFile }), /different input\/plan digest|generated candidate file was changed/);
});

test('failed campaign trials retain reflection, proposed action, and frozen gates', async (t) => {
  const { dir, planFile } = await fixture(t);
  const plan = JSON.parse(await fs.readFile(planFile));
  plan.proposal.strategy = 'reflective';
  plan.proposal.parameters.push({ path: 'hyper.batchSize', values: [2, 4, 8] });
  plan.budget.maxTrials = 2;
  await fs.writeFile(planFile, JSON.stringify(plan));
  const configFile = path.join(dir, 'base.json');
  const config = JSON.parse(await fs.readFile(configFile));
  config.eval.gates[0].threshold = -1;
  await fs.writeFile(configFile, JSON.stringify(config));
  const run = await runCampaign({ planFile });
  assert.equal(run.result.status, 'no_candidate_passed');
  assert.deepEqual(run.result.trials.map((item) => item.id), ['p-1-1', 'p-0-0']);
  assert.equal(run.result.trials[0].reflection.observation.failedGates[0].threshold, -1);
  assert.equal(run.result.trials[0].reflection.action.method, 'contrast');
  assert.equal(run.result.trials[1].proposal.parentId, 'p-1-1');
  assert.equal(run.result.trials[1].stepsCharged, 2);
  assert.equal((await runCampaign({ planFile })).replayed, true);
});

test('reflective campaign completes in a standalone background worker', async (t) => {
  const { dir, planFile } = await fixture(t);
  const plan = JSON.parse(await fs.readFile(planFile));
  plan.proposal.strategy = 'reflective';
  plan.proposal.parameters.push({ path: 'hyper.batchSize', values: [2, 4, 8] });
  await fs.writeFile(planFile, JSON.stringify(plan));
  const configFile = path.join(dir, 'base.json');
  const config = JSON.parse(await fs.readFile(configFile));
  config.eval.gates[0].threshold = -1;
  await fs.writeFile(configFile, JSON.stringify(config));
  const priorLauncher = process.env.TMF_BACKGROUND_LAUNCHER;
  process.env.TMF_BACKGROUND_LAUNCHER = 'direct';
  t.after(() => { if (priorLauncher === undefined) delete process.env.TMF_BACKGROUND_LAUNCHER;
    else process.env.TMF_BACKGROUND_LAUNCHER = priorLauncher; });
  const { startCampaign, backgroundCampaignStatus } = await import('../src/controller/background.mjs');
  const started = await startCampaign({ planFile });
  assert.equal(started.state, 'starting');
  let final;
  // The full suite runs workers concurrently on shared CI runners. Bound the
  // wait by wall clock, but leave enough room for scheduler contention.
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    final = await backgroundCampaignStatus({ planFile });
    if (['succeeded', 'failed', 'interrupted'].includes(final.state)) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  assert.equal(final.state, 'succeeded', JSON.stringify({ state: final.run, worker: final.workerResult }));
  assert.equal(final.decision.status, 'no_candidate_passed');
  assert.equal(final.decision.trials[0].reflection.action.nextCandidateId, 'p-0-0');
  assert.equal(final.decision.trials[1].proposal.parentId, 'p-1-1');
  assert.equal(final.decision.exported, false);
});
