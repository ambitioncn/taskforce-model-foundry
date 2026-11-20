// Contract test for docs/acceptance/r1-r7-reconciliation.md: the reconciliation
// report is acceptance evidence, so the numbers it quotes must stay tied to the
// code and to the honest project state. Same shape as tutorial-docs.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { exampleConfig } from '../src/config.mjs';
import { assertReportIntegrity } from '../src/jobs/evaluate.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const doc = await fs.readFile(
  path.join(root, 'docs/acceptance/r1-r7-reconciliation.md'),
  'utf8',
);

test('every requirement R1..R7 has its own section', () => {
  for (const r of ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']) {
    assert.match(doc, new RegExp(`^## ${r} — `, 'm'), `missing section for ${r}`);
  }
});

test('the summary table has one verdict row per requirement', () => {
  const verdicts = new Map();
  for (const line of doc.split('\n')) {
    const m = /^\| (R[1-7]) [^|]*\| (met|partial|blocked) \|/.exec(line);
    if (m) verdicts.set(m[1], m[2]);
  }
  assert.equal(verdicts.size, 7, `expected 7 verdict rows, got ${verdicts.size}`);
});

function tableVerdict(req) {
  for (const line of doc.split('\n')) {
    const m = new RegExp(`^\\| ${req} [^|]*\\| (met|partial|blocked) \\|`).exec(line);
    if (m) return m[1];
  }
  return undefined;
}

test('quoted hyperparameters match the shipped example config', () => {
  const { hyper } = exampleConfig();
  assert.equal(hyper.lr, 0.005);
  assert.match(doc, new RegExp(`hyper=\\{steps:${hyper.steps}, lr:${hyper.lr}, batchSize:${hyper.batchSize}\\}`));
  // cp62/cp63 lineage was recorded at the 2400-step budget with the same lr.
  assert.match(doc, new RegExp(`steps 2400 / lr ${hyper.lr} / batchSize ${hyper.batchSize}`));
});

test('the stated project_completion agrees with the real-gap list', () => {
  const blocked = numberedBlock(BLOCKED_HEADER, '  ');
  if (blocked.length > 0) {
    assert.match(doc, /project_completion=false/, 'cannot claim completion while real gaps remain');
    assert.ok(!/project_completion=true/.test(doc), 'gap list is non-empty, so completion must read false');
    assert.match(doc, /未验收/);
  } else {
    assert.match(doc, /project_completion=true/, 'gap list is empty, so completion must be stated');
    assert.ok(!/project_completion=false/.test(doc), 'stale project_completion=false left in the report');
  }
  // Gate thresholds are quoted verbatim and must never be relaxed in the docs.
  for (const t of ['>= 0.3', '<= 1.5', '>= 0.25', '>= 0.1', '>= -0.05']) {
    assert.ok(doc.includes(t), `missing gate threshold ${t}`);
  }
});

// cp79 pinned the closing "剩余工作" list to the "未验收 / blocked" list above
// it. cp85 corrected what each list is for: the completion contract forbids
// production-host changes, commit/push and npm publish outright and requires
// separate authorization, so they are out-of-contract actions, not R1–R7
// deliverables — they must be recorded honestly but must never be encoded as
// a precondition for closing the project. They now live in their own two
// lists, pinned to the same three gates so neither can silently drop one.
function authorizationGates(block) {
  const gates = [];
  for (const raw of block) {
    const item = raw.trim();
    const tags = [];
    if (item.includes('npm publish')) tags.push('publish');
    if (/commit\/push|push 到/.test(item)) tags.push('push');
    if (item.includes('生产 OpenClaw')) tags.push('production-host');
    assert.equal(tags.length, 1, `item must name exactly one gate: ${item}`);
    gates.push(tags[0]);
  }
  return gates;
}

// cp90: the in-section gap list legitimately became empty once cp87/cp88 closed
// the last R7 deliverable gap, so its header can no longer carry the pending
// markers ('未验收' / 'blocked') that drive the per-section verdict test below.
// The header must still exist (see numberedBlock) so an emptied list stays
// visible rather than silently disappearing.
const BLOCKED_HEADER = '**R7 合同交付项内的真实缺口';

function numberedBlock(after, indent) {
  const lines = doc.split('\n');
  const start = lines.findIndex((l) => l.includes(after));
  assert.ok(start >= 0, `missing list header: ${after}`);
  const item = new RegExp(`^${indent}\\d+\\. `);
  const block = [];
  for (const line of lines.slice(start + 1)) {
    if (!item.test(line)) break;
    block.push(line);
  }
  return block;
}

test('both out-of-contract lists name the same three separately-authorized actions', () => {
  const inSection = authorizationGates(numberedBlock('**合同外动作（完成合同原文明文禁止', '  '));
  const closing = authorizationGates(numberedBlock('合同外动作（**不构成本项目关闭前提**', ''));
  assert.deepEqual(inSection, ['production-host', 'push', 'publish']);
  assert.deepEqual(closing, ['production-host', 'push', 'publish']);
  assert.deepEqual(closing, inSection);
  // publish has only ever been exercised as a dry run; say so.
  assert.match(doc, /npm pack --dry-run/);
});

// The correction only holds if the report keeps saying *why* those three are
// excluded from the verdict. Without this, a later edit could quietly fold
// them back into the blocking set.
test('the out-of-contract actions are stated not to gate project closure', () => {
  assert.match(doc, /不构成 R1–R7 验收或项目关闭前提/);
  assert.match(doc, /不构成本项目关闭前提/);
  for (const gate of ['需单独授权', '未执行']) {
    assert.ok(doc.includes(gate), `missing honest out-of-contract wording: ${gate}`);
  }
});

// The blocked list must hold only real R7 deliverable gaps, and none of the
// three out-of-contract actions may reappear there — that is exactly the
// mis-coupling cp84 found.
test('the blocked list holds real R7 gaps, not the out-of-contract actions', () => {
  const blocked = numberedBlock(BLOCKED_HEADER, '  ');
  assert.ok(blocked.length > 0 || tableVerdict('R7') === 'met', 'blocked list and R7 verdict disagree');
  const block = blocked.join('\n');
  for (const forbidden of ['npm publish', 'commit/push', '生产 OpenClaw']) {
    assert.ok(!block.includes(forbidden), `out-of-contract action must not gate R7: ${forbidden}`);
  }
  // While a gap is listed it must say it is pending execution, not pending
  // authorization; once empty that wording goes with the item it described.
  if (blocked.length > 0) assert.match(sectionBody('R7').body, /属待执行而非待授权/);
});

test('the closing remaining-work list mirrors the blocked gaps', () => {
  const blocked = numberedBlock(BLOCKED_HEADER, '  ');
  const remaining = numberedBlock('剩余工作（与上文', '');
  assert.equal(remaining.length, blocked.length, 'remaining-work list dropped a real gap');
  if (blocked.length > 0) {
    assert.ok(remaining.join('\n').includes('宿主'), 'remaining work must name the host-discovery gap');
  }
});

test('the verdict sentence names the real remaining R7 gap', () => {
  const [sentence] = doc.match(/^R7 的本地\/隔离环境部分.*$/m) ?? [];
  assert.ok(sentence, 'missing the R7 verdict sentence');
  const blocked = numberedBlock(BLOCKED_HEADER, '  ');
  if (blocked.length > 0) {
    assert.match(sentence, /真实缺口未验收/, 'verdict must name the outstanding gap');
  } else {
    // The sentence may only read closed if it cites the receipts that closed it.
    assert.match(sentence, /cp87/, 'closed verdict must cite the closing receipt');
    assert.ok(!/真实缺口未验收/.test(sentence), 'stale gap claim with an empty gap list');
  }
  assert.ok(!/npm publish|commit\/push/.test(sentence), 'verdict sentence must not gate on out-of-contract actions');
});

// cp80 pinned the two gate lists to each other, but the summary table was still
// free to drift from them: editing only the R7 row to "met" would have let the
// report read as accepted while the three human-authorization gates below it
// were still listed as blocked. Derive the expected R7 verdict from that list
// instead of hard-coding it, so the table and the gates move together in both
// directions.
test('the R7 verdict tracks the blocked gate list in both directions', () => {
  const blocked = numberedBlock(BLOCKED_HEADER, '  ');
  const expected = blocked.length > 0 ? 'partial' : 'met';
  assert.equal(
    tableVerdict('R7'),
    expected,
    blocked.length > 0
      ? `R7 must stay partial while ${blocked.length} gate(s) are blocked: ${blocked.length}`
      : 'R7 may only read met once the blocked gate list is empty',
  );
});

// cp81 pinned only the R7 row. The other six rows were still free to drift
// from the prose below them: a row could read "met" while its own section
// still said 待人工授权, and a stale "partial" row could survive after its
// section stopped listing any pending gate. Derive each verdict from its own
// section instead, in both directions.
const PENDING_MARKERS = ['待人工授权', '未验收', 'blocked'];

function sectionBody(req) {
  const lines = doc.split('\n');
  const start = lines.findIndex((l) => l.startsWith(`## ${req} — `));
  assert.ok(start >= 0, `missing section for ${req}`);
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((l) => l.startsWith('## '));
  return { heading: lines[start], body: (end < 0 ? rest : rest.slice(0, end)).join('\n') };
}

test('each summary verdict tracks the pending markers in its own section', () => {
  for (const req of ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']) {
    const { heading, body } = sectionBody(req);
    const hit = PENDING_MARKERS.filter((m) => body.includes(m));
    const expected = hit.length > 0 ? 'partial' : 'met';
    assert.equal(
      tableVerdict(req),
      expected,
      hit.length > 0
        ? `${req} must stay partial while its section still says: ${hit.join(', ')}`
        : `${req} may only read ${tableVerdict(req)} if its section names a pending gate`,
    );
    // The heading carries the same verdict, so it cannot be relaxed on its own.
    assert.equal(
      heading.includes('（partial）'),
      expected === 'partial',
      `${req} heading and verdict disagree: ${heading}`,
    );
  }
});

// cp83: the R4 numbers were still only document literals — the report could
// quote any loss it liked. Pin them to the machine-generated evaluate report
// itself, kept in-repo byte-identical to the cp62 artifact, and let the
// shipped integrity check reject a doctored fixture so the coupling cannot be
// satisfied by editing both sides.
const evalReport = assertReportIntegrity(
  JSON.parse(await fs.readFile(path.join(root, 'docs/acceptance/evidence/cp62-eval-report.json'), 'utf8')),
  { jobId: 'cp62-lr005' },
);

test('the pinned cp62 eval report is the one the R4 section cites', () => {
  assert.equal(evalReport.lineage.jobId, 'cp62-lr005');
  assert.equal(evalReport.verdict.status, 'passed');
  assert.equal(evalReport.gates.passed, true);
});

test('quoted R4 metrics match the pinned eval report', () => {
  const { body } = sectionBody('R4');
  const tuned = evalReport.metrics.tuned.holdout;
  const baseline = evalReport.metrics.baseline.holdout;
  assert.ok(body.includes(`loss **${tuned.loss}**`), `R4 must quote tuned loss ${tuned.loss}`);
  assert.ok(body.includes(`perplexity ${tuned.perplexity}`), `R4 must quote perplexity ${tuned.perplexity}`);
  assert.ok(body.includes(`baseline **${baseline.loss}**`), `R4 must quote baseline loss ${baseline.loss}`);
  assert.ok(
    body.includes(`独立 holdout ${evalReport.independence.holdout} 条`),
    `R4 must quote the holdout size ${evalReport.independence.holdout}`,
  );
  // The summary row carries the same two numbers, so it cannot drift either.
  const row = doc.split('\n').find((l) => /^\| R4 /.test(l));
  assert.ok(row.includes(String(tuned.loss)) && row.includes(String(baseline.loss)), `R4 row drifted: ${row}`);
});

test('quoted R4 gate thresholds and actuals match the pinned eval report', () => {
  const { body } = sectionBody('R4');
  const results = evalReport.gates.results;
  const comparators = { gte: '>=', lte: '<=' };
  assert.ok(
    body.includes(`门限 ${results.filter((g) => g.ok).length}/${results.length} 通过`),
    `R4 must quote the real gate tally ${results.filter((g) => g.ok).length}/${results.length}`,
  );
  for (const gate of results) {
    assert.ok(
      body.includes(`\`${gate.id} ${comparators[gate.comparator]} ${gate.threshold}\` → ${gate.actual}`),
      `R4 must quote ${gate.id} ${comparators[gate.comparator]} ${gate.threshold} → ${gate.actual}`,
    );
  }
  assert.ok(
    body.includes(`\`status=${evalReport.verdict.status}\``) &&
      body.includes(`\`improvementClaimAllowed=${evalReport.verdict.improvementClaimAllowed}\``) &&
      body.includes(`\`deliveryEligible=${evalReport.verdict.deliveryEligible}\``),
    'R4 must quote the report verdict verbatim',
  );
});
