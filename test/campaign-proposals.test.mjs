import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { synthesize } from '../src/api.mjs';
import { buildProposalPool, chooseNextProposal } from '../src/controller/index.mjs';
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
