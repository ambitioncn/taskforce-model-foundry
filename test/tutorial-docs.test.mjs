// Contract test for docs/tutorial.md, same shape as integration-docs.test.mjs:
// the tutorial's tables are parsed and compared against the code they describe,
// so a mode, a checkpoint name or an export error can't drift out of the docs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { MODES } from '../src/config.mjs';
import { assertReportIntegrity } from '../src/jobs/evaluate.mjs';
import { EVAL_REPORT_FORMAT } from '../src/eval/evaluator.mjs';
import { sha256Json } from '../src/core/hash.mjs';
import { loadDefaultBackends, listBackends } from '../src/train/backends/index.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const doc = await fs.readFile(path.join(root, 'docs/tutorial.md'), 'utf8');

function rows(headingPrefix, after) {
  const body = doc.slice(doc.indexOf(after));
  const out = [];
  for (const line of body.split('\n')) {
    if (!line.startsWith('|')) {
      if (out.length) break;
      continue;
    }
    const cells = line.split('|').slice(1, -1).map((c) => c.trim());
    if (cells.every((c) => /^-+$/.test(c)) || cells[0] === headingPrefix) continue;
    out.push(cells);
  }
  return out;
}

const code = (s) => s.replace(/`/g, '');

test('tutorial mode table matches MODES and the checkpoint the train job writes', () => {
  const table = rows('mode', '## 10. Full-parameter runs');
  assert.deepEqual(table.map((r) => code(r[0])).sort(), [...MODES].sort());
  for (const [mode, , ckpt] of table) {
    // src/jobs/train.mjs: `const isAdapter = mode === 'lora'` →
    // model/adapter.json for lora, model/tuned.json otherwise.
    const expected = code(mode) === 'lora' ? 'adapter.json' : 'tuned.json';
    assert.equal(code(ckpt), expected, `${mode}: checkpoint name`);
  }
});

test('tutorial backend table matches each backend capabilities.modes', async () => {
  await loadDefaultBackends();
  const actual = new Map(listBackends().map((b) => [b.id, b.capabilities.modes]));
  const table = rows('backend', '| `builtin-js` |');
  assert.ok(table.length > 0, 'backend table parsed');
  assert.deepEqual(table.map((r) => code(r[0])).sort(), [...actual.keys()].sort());
  for (const [id, modes] of table) {
    const documented = modes.split(',').map((m) => code(m).trim());
    assert.deepEqual(documented, actual.get(code(id)), `${id}: modes`);
  }
});

test('tutorial export --report error table matches assertReportIntegrity', () => {
  const collect = (input, opts) => {
    try {
      assertReportIntegrity(input, opts);
      return null;
    } catch (err) {
      return err.message;
    }
  };
  // Every documented row must correspond to a message the function really
  // throws, so all five are produced here from real inputs.
  const wellFormed = { format: EVAL_REPORT_FORMAT, lineage: { jobId: 'job-a' } };
  const digest = sha256Json({ ...wellFormed, reportDigest: undefined });
  const thrown = [
    collect(null),
    collect({ format: 'not-the-format' }),
    collect({ format: EVAL_REPORT_FORMAT }),
    collect({ ...wellFormed, reportDigest: 'deadbeef' }),
    collect({ ...wellFormed, reportDigest: digest }, { jobId: 'job-b' }),
  ];
  assert.equal(thrown.filter(Boolean).length, 5, 'all five failure modes actually throw');
  // Sanity: the same report with the matching jobId is accepted, so row 5 is
  // about lineage and not about a still-broken digest.
  assert.equal(collect({ ...wellFormed, reportDigest: digest }, { jobId: 'job-a' }), null);

  const table = rows('situation', '| not a JSON object |');
  assert.equal(table.length, 5, 'five documented failure modes');
  const toRegex = (s) => new RegExp('^' + code(s)
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    .replace(/<[a-zA-Z]+>/g, '.+') + '$');
  for (const [situation, message] of table) {
    const re = toRegex(message);
    assert.ok(
      thrown.some((t) => t && re.test(t)),
      `documented error has no matching thrown message (${situation}): ${message}`,
    );
  }
});

test('tutorial hyperparameter table matches the config tmf init writes', async () => {
  const { exampleConfig } = await import('../src/config.mjs');
  const table = rows('key', '### Default hyperparameters');
  const documented = Object.fromEntries(table.map(([k, v]) => [code(k), Number(code(v))]));
  assert.deepEqual(documented, exampleConfig().hyper);
  // Regression guard for the lr that actually converged on the builtin-js
  // AdamW path: 0.05 oscillates and the holdout improvement is lost, so the
  // scaffold default must stay at the small end instead of drifting back up.
  assert.equal(exampleConfig().hyper.lr, 0.005);
  assert.ok(exampleConfig().hyper.lr <= 0.005, 'default lr must not regress upward');
});

test('tutorial report paths use the real runRoot default, not runs/', async () => {
  const { exampleConfig, DEFAULTS } = await import('../src/config.mjs');
  const runRoot = (DEFAULTS ?? exampleConfig()).runRoot ?? exampleConfig().runRoot;
  assert.equal(runRoot, '.tmf-run', 'runRoot default moved; the tutorial paths need updating');
  // The walkthrough in a fresh install showed a copied `runs/<jobId>` path
  // never resolves, so no documented --report argument may use that prefix.
  assert.ok(
    !/--report\s+runs\//.test(doc),
    'tutorial documents a --report path under runs/, which runRoot never produces',
  );
  for (const m of doc.matchAll(/--report\s+(\S+)/g)) {
    assert.ok(
      m[1].startsWith(`${runRoot}/`),
      `documented --report path must live under ${runRoot}/: ${m[1]}`,
    );
  }
  // Export resolves the job id from the base directory name when --job is
  // absent, so every documented export that pins a report must pass --job.
  for (const line of doc.split('\n')) {
    if (!/\btmf export\b/.test(line) || !/--report/.test(line)) continue;
    assert.ok(/--job\s/.test(line), `documented export pins a report without --job: ${line.trim()}`);
  }
  assert.ok(
    /exits 1 when the gates fail, yet it\s*\n?\s*still writes `eval-report\.json`/.test(doc),
    'tutorial must state that a failing evaluate still writes the report',
  );
});
