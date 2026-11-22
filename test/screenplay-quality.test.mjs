import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { sha256File } from '../src/core/hash.mjs';
import { SCREENPLAY_PROFILE, SCREENPLAY_GATES } from '../src/eval/screenplay-profile.mjs';
import { createScreenplayQualityContract, prepareScreenplayQuality, finalizeScreenplayQuality, screenplayScriptDigest } from '../src/eval/screenplay-quality.mjs';

test('the real G5 scripts score 120 items, build blind pairs and seal a fully gated report', async (t) => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-screenplay-quality-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  const suiteFile = path.join(dir, 'suite.json');
  const baseOutputs = path.join(dir, 'base.jsonl');
  const daptOutputs = path.join(dir, 'dapt.jsonl');
  const candidateOutputs = path.join(dir, 'candidate.jsonl');
  const cleanDaptReport = path.join(dir, 'dapt-loss.json');
  const cleanCandidateReport = path.join(dir, 'candidate-loss.json');
  const tiers = [...Array(85).fill('scene'), ...Array(25).fill('broad'), ...Array(10).fill('long')];
  const taskTypes = [
    ...Array(20).fill('short_drama_hook'), ...Array(15).fill('film_tv_scene'),
    ...Array(15).fill('dialogue_subtext'), ...Array(15).fill('action_blocking_silent'),
    ...Array(15).fill('hard_constraints'), ...Array(10).fill('synopsis'),
    ...Array(10).fill('treatment_three_act'), ...Array(10).fill('episode_outline'),
    ...Array(5).fill('character_bible'), ...Array(5).fill('rewrite_continue_compress'),
  ];
  const items = tiers.map((tier, i) => ({
    id: `item-${String(i).padStart(3, '0')}`, prompt: `请写剧本 ${i}`, task_type: taskTypes[i], length_tier: tier,
    max_new_tokens: 10, required: ['钥匙'], forbidden: [], target_length_chars: [1, 100],
  }));
  await fs.writeFile(suiteFile, JSON.stringify({ items }));
  const make = (candidate) => items.map((item) => ({
    id: item.id, prompt: item.prompt,
    output: `内景 房间 日\n甲：${candidate ? '钥匙在这里。' : '你好。'}\n乙：好。`,
    new_tokens: 5, max_new_tokens: 10, hit_max_new_tokens: false,
  }));
  for (const [file, rows] of [[baseOutputs, make(false)], [daptOutputs, make(false)], [candidateOutputs, make(true)]]) {
    await fs.writeFile(file, rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
  }
  const loss = (name) => ({ base_model: 'frozen-base', val_jsonl: 'frozen-clean-val', samples: 64, max_length: 512,
    quantization: 'nf4', tokenizer: { bos_token_id: 2, eos_token_id: 1 },
    model: { class: 'FixtureLM', attn_implementation: 'sdpa' },
    results: [{ model: name, protocol: 'chat_wrapped', samples: 64, token_weighted_loss: 1.0 }] });
  await fs.writeFile(cleanDaptReport, JSON.stringify(loss('dapt')));
  await fs.writeFile(cleanCandidateReport, JSON.stringify(loss('candidate')));
  const contract = {
    profile: SCREENPLAY_PROFILE, suiteSha256: await sha256File(suiteFile),
    scorerSha256: await screenplayScriptDigest(), gates: SCREENPLAY_GATES,
    references: { baseOutputs: await sha256File(baseOutputs), daptOutputs: await sha256File(daptOutputs), cleanDaptReport: await sha256File(cleanDaptReport) },
  };
  assert.deepEqual(await createScreenplayQualityContract({ suiteFile, baseOutputs, daptOutputs, cleanDaptReport }), contract);
  const prepared = await prepareScreenplayQuality({
    contract, jobId: 'fixture-job', evalReport: { reportDigest: 'a'.repeat(64), lineage: { tunedFingerprint: 'b'.repeat(64) } },
    checkpointSha256: 'c'.repeat(64), suiteFile, baseOutputs, daptOutputs, candidateOutputs,
    cleanDaptReport, cleanDaptModel: 'dapt', outDir: path.join(dir, 'stage'),
  });
  const key = JSON.parse(await fs.readFile(path.join(prepared.blindDir, 'key.json'), 'utf8'));
  const verdicts = [];
  for (const judge of ['judge1', 'judge2']) {
    const file = path.join(dir, `${judge}.jsonl`);
    const rows = [];
    for (const [id, pair] of Object.entries(key.pairs)) for (const order of ['order1', 'order2']) {
      rows.push({ id, order, verdict: pair[order].A === 'candidate' ? 'A' : 'B' });
    }
    await fs.writeFile(file, rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
    verdicts.push(`${judge}=${file}`);
  }
  const result = await finalizeScreenplayQuality({ stageFile: prepared.stageFile, contract, judgeVerdicts: verdicts,
    cleanCandidateReport, cleanCandidateModel: 'candidate', reportFile: path.join(dir, 'quality.json') });
  assert.equal(result.report.gates.passed, true);
  assert.equal(result.report.metrics.quality.sceneCount, 85);
  assert.equal(result.report.metrics.quality.broadCount, 35);
  assert.equal(result.report.metrics.quality.abMinJudged, 120);
  assert.equal(result.report.metrics.quality.abJudgeCount, 2);
  await assert.rejects(() => finalizeScreenplayQuality({ stageFile: prepared.stageFile, contract, judgeVerdicts: verdicts.slice(0, 1),
    cleanCandidateReport, cleanCandidateModel: 'candidate', reportFile: path.join(dir, 'partial.json') }), /two independent/);
  await fs.writeFile(cleanCandidateReport, JSON.stringify({ ...loss('candidate'), tokenizer: { bos_token_id: 9 } }));
  await assert.rejects(() => finalizeScreenplayQuality({ stageFile: prepared.stageFile, contract, judgeVerdicts: verdicts,
    cleanCandidateReport, cleanCandidateModel: 'candidate', reportFile: path.join(dir, 'wrong-tokenizer.json') }), /tokenizer/);
  await fs.writeFile(cleanCandidateReport, JSON.stringify(loss('candidate')));
  const weaker = { ...contract, gates: contract.gates.map((gate) => gate.id === 'ab_win' ? { ...gate, threshold: 0.5 } : gate) };
  await assert.rejects(() => prepareScreenplayQuality({ contract: weaker }), /original thresholds/);
  await fs.appendFile(candidateOutputs, '\n');
  await assert.rejects(() => finalizeScreenplayQuality({ stageFile: prepared.stageFile, contract, judgeVerdicts: verdicts,
    cleanCandidateReport, cleanCandidateModel: 'candidate', reportFile: path.join(dir, 'stale.json') }), /source changed/);
});
