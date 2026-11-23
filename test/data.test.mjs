import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp, writeFile, mkdir } from 'node:fs/promises';
import { cleanText, canonicalKey, qualityReport } from '../src/data/clean.mjs';
import { scanPii, redactPii, applyPiiPolicy } from '../src/data/pii.mjs';
import { NearDuplicateIndex, jaccard, exactKey, minhashSignature, signatureSimilarity } from '../src/data/dedupe.mjs';
import { ingestAll, parseDelimited } from '../src/data/ingest.mjs';
import { normalizeDatasetDescriptor, summarizeProvenance, assertTrainableClassification } from '../src/data/provenance.mjs';
import { splitRecords, checkLeakage, assertNoLeakage, MIN_SCORABLE_HOLDOUT } from '../src/data/split.mjs';
import { runDataPipeline } from '../src/data/pipeline.mjs';
import { writeSyntheticCorpus, generateTaskRecords, expectedRouting } from '../src/data/synth.mjs';
import { readJsonl } from '../src/core/fsx.mjs';
import { Rng } from '../src/core/rng.mjs';

async function tmp(prefix = 'tmf-data-') {
  return mkdtemp(path.join(os.tmpdir(), prefix));
}

const SECRETS = ['ops.lead@contoso.example', '13800138000', 'sk-abcdefghijklmnopqrstuvwx', '4111111111111111'];
const ZWSP = String.fromCharCode(0x200b);
const CONTROL = String.fromCharCode(1, 2, 3);

test('cleanText normalises whitespace, unicode and control characters', () => {
  const noisy = `  Ticket:\tRefund  request \n\n\n next ${ZWSP} `;
  const { text, applied } = cleanText(noisy);
  assert.equal(text, 'Ticket: Refund request\n\nnext');
  assert.ok(applied.includes('control_chars'));
  assert.ok(applied.includes('whitespace'));
  assert.ok(applied.includes('trim'));
  assert.equal(cleanText('ABC', { lowercase: true }).text, 'abc');
});

test('canonicalKey ignores case and punctuation differences', () => {
  assert.equal(canonicalKey('Hello, World!!'), canonicalKey('  hello   world  '));
  assert.notEqual(canonicalKey('billing refund'), canonicalKey('billing invoice'));
});

test('qualityReport rejects short, repetitive, placeholder and low-signal text', () => {
  assert.ok(!qualityReport('hi').ok);
  assert.match(qualityReport('hi').reasons[0], /too_short/);
  assert.match(qualityReport('ab'.repeat(40)).reasons.join(','), /repetitive/);
  assert.ok(!qualityReport('lorem ipsum dolor sit amet body text').ok);
  assert.ok(!qualityReport('x'.repeat(5000)).ok);
  assert.ok(!qualityReport('!!! ??? ... ,,, ;;; ::: --- +++').ok, 'punctuation-only text has a low alnum ratio');
  assert.ok(!qualityReport('').ok);
  const good = qualityReport('the quarterly invoice was approved by finance on monday');
  assert.ok(good.ok);
  assert.ok(good.score > 0.4 && good.score <= 1);
});

test('non-printable characters are counted without a stateful regex bug', () => {
  const report = qualityReport(`ab${CONTROL}cd`, { minChars: 1 });
  assert.ok(report.metrics.nonPrintableRatio > 0.4, `expected a high ratio, got ${report.metrics.nonPrintableRatio}`);
  assert.match(report.reasons.join(','), /non_printable/);
});

test('sensitive information is detected across all supported kinds', () => {
  const text = [
    'contact ops.lead@contoso.example',
    'phone 13800138000',
    'token sk-abcdefghijklmnopqrstuvwx',
    'card 4111111111111111',
    'host 10.0.0.7',
    'api_key: hunter2secret',
    'id 11010519491231002X',
  ].join(' | ');
  const kinds = new Set(scanPii(text).map((f) => f.kind));
  for (const expected of ['email', 'phone', 'credential', 'bank_card', 'ip', 'cn_id']) {
    assert.ok(kinds.has(expected), `expected to detect ${expected}`);
  }
  assert.equal(scanPii('route subject billing refund routine').length, 0);
});

test('redaction removes the sensitive value and keeps the surrounding text', () => {
  const { text } = redactPii('mail ops@corp.example about card 4111111111111111 today');
  assert.match(text, /\[REDACTED:EMAIL\]/);
  assert.match(text, /\[REDACTED:CARD\]/);
  assert.ok(!text.includes('ops@corp.example'));
  assert.ok(!text.includes('4111111111111111'));
  assert.match(text, /^mail .* about card .* today$/);
});

test('card and national id detection validate checksums instead of matching digits', () => {
  assert.equal(scanPii('card 4111111111111112').filter((f) => f.kind === 'bank_card').length, 0, 'a failing Luhn check is not a card');
  assert.equal(scanPii('card 4111111111111111').filter((f) => f.kind === 'bank_card').length, 1);
  assert.equal(scanPii('id 11010519491231002A').filter((f) => f.kind === 'cn_id').length, 0);
  assert.equal(scanPii('id 11010519491231002X').filter((f) => f.kind === 'cn_id').length, 1);
});

test('pii policies keep, redact, drop or fail as configured', () => {
  assert.equal(applyPiiPolicy('route subject billing refund routine').action, 'keep');
  const dirty = 'owner ops@corp.example handles the billing refund';
  assert.equal(applyPiiPolicy(dirty, { policy: 'redact' }).action, 'redact');
  assert.equal(applyPiiPolicy(dirty, { policy: 'drop' }).action, 'drop');
  // 'fail' aborts only at or above blockSeverity, which defaults to critical:
  // an address is redacted, a credential stops the job.
  assert.doesNotThrow(() => applyPiiPolicy(dirty, { policy: 'fail' }));
  assert.doesNotThrow(() => applyPiiPolicy('host 10.0.0.7 is up', { policy: 'fail' }));
  assert.throws(() => applyPiiPolicy('token sk-abcdefghijklmnopqrstuvwx leaked', { policy: 'fail' }), /pii policy "fail"/);
  assert.throws(() => applyPiiPolicy(dirty, { policy: 'fail', blockSeverity: 'high' }), /pii policy "fail"/);
});

test('near-duplicate index finds typo re-submissions and ignores unrelated text', () => {
  const long = 'the finance team reviewed the billing refund escalation this week and the compliance team documented the legal audit';
  const index = new NearDuplicateIndex();
  index.add('a', long);
  const hit = index.query(long.replace('documented', 'documentd'));
  assert.ok(hit.match, 'a one-character typo should still be a near duplicate');
  assert.ok(hit.match.similarity >= 0.85);
  assert.equal(index.query('deploy the hermes gateway to staging tonight').match, null);
  assert.equal(index.size, 1);
});

test('minhash similarity approximates jaccard', () => {
  const a = 'the platform team escalated the outage incident before the audit window closed';
  const b = 'the platform team escalated the outage incident before the audit window closes';
  const approx = signatureSimilarity(minhashSignature(a), minhashSignature(b));
  const exact = jaccard(a, b);
  assert.ok(Math.abs(approx - exact) < 0.2, `approx=${approx} exact=${exact}`);
  assert.equal(exactKey('Hello  World!'), exactKey('hello world'));
});

test('dedupe index rejects an inconsistent band configuration', () => {
  assert.throws(() => new NearDuplicateIndex({ numHashes: 64, bands: 7 }), /must be divisible/);
});

test('ingestion reads jsonl, csv, tsv, json and text with provenance attached', async () => {
  const dir = await tmp();
  await writeFile(path.join(dir, 'a.jsonl'), '{"prompt":"p1","completion":"c1"}\n{"text":"free form document"}\nnot-json\n');
  await writeFile(path.join(dir, 'b.csv'), 'prompt,completion\n"route x","queue=finance"\n');
  await writeFile(path.join(dir, 'c.tsv'), 'text\nthe tab separated document\n');
  await writeFile(path.join(dir, 'd.json'), '[{"text":"from a json array"}]');
  await writeFile(path.join(dir, 'e.txt'), 'first document\n\nsecond document\n');
  const result = await ingestAll([{ name: 'mixed', path: dir, license: 'CC0-1.0' }], { baseDir: dir });
  const texts = result.records.map((r) => r.text);
  for (const expected of ['p1c1', 'free form document', 'route xqueue=finance', 'the tab separated document', 'from a json array', 'first document', 'second document']) {
    assert.ok(texts.includes(expected), `expected to ingest "${expected}"`);
  }
  assert.equal(result.errors.length, 1, 'the malformed jsonl line is reported, not silently dropped');
  for (const record of result.records) {
    assert.equal(record.source.license, 'CC0-1.0');
    assert.ok(record.source.file && record.source.rawSha256);
  }
});

test('ingestion surfaces a missing path and an unusable record shape', async () => {
  await assert.rejects(ingestAll([{ name: 'gone', path: '/definitely/not/here', license: 'CC0-1.0' }]), /path is unreadable/);
  const dir = await tmp();
  await writeFile(path.join(dir, 'x.jsonl'), '{"question":"no usable field"}\n');
  const result = await ingestAll([{ name: 'bad', path: dir, license: 'CC0-1.0' }], { baseDir: dir });
  assert.equal(result.records.length, 0);
  assert.match(result.errors[0].reason, /missing usable fields/);
  await assert.rejects(ingestAll([]), /at least one dataset descriptor/);
});

test('delimited parser handles quotes, embedded delimiters and newlines', () => {
  const rows = parseDelimited('a,b\n"x,1","he said ""hi"""\n"multi\nline",2\n', ',');
  assert.deepEqual(rows[1], ['x,1', 'he said "hi"']);
  assert.deepEqual(rows[2], ['multi\nline', '2']);
});

test('dataset descriptors require a licence and a known classification', () => {
  assert.throws(() => normalizeDatasetDescriptor({ path: 'x' }), /missing a license/);
  assert.throws(() => normalizeDatasetDescriptor({ license: 'MIT' }), /missing a path/);
  assert.throws(() => normalizeDatasetDescriptor({ path: 'x', license: 'MIT', classification: 'secret' }), /unknown classification/);
  assert.equal(normalizeDatasetDescriptor({ path: 'x', license: 'MIT' }).classification, 'internal');
});

test('restricted classifications are refused for training', () => {
  const records = [
    { source: { dataset: 'ok', classification: 'internal', license: 'MIT', file: 'a' } },
    { source: { dataset: 'locked', classification: 'restricted', license: 'MIT', file: 'b' } },
  ];
  assert.throws(() => assertTrainableClassification(records), /not permitted for training/);
  assert.doesNotThrow(() => assertTrainableClassification([records[0]]));
  const summary = summarizeProvenance(records);
  assert.equal(summary.totalRecords, 2);
  assert.deepEqual(summary.licenses, ['MIT']);
  assert.deepEqual(summary.classifications, ['internal', 'restricted']);
});

test('splits are deterministic, order-independent and ratio-checked', () => {
  const records = Array.from({ length: 300 }, (_, i) => ({ id: `r${i}`, text: `ticket ${i} about the billing refund case ${i}` }));
  const a = splitRecords(records);
  const b = splitRecords(records.slice().reverse());
  const ids = (s) => [...new Set(s.map((r) => r.id))].sort();
  for (const name of ['train', 'validation', 'holdout']) {
    assert.deepEqual(ids(a[name]), ids(b[name]), `${name} must not depend on input order`);
  }
  assert.ok(a.train.length > a.validation.length && a.holdout.length > 0);
  assert.throws(() => splitRecords(records, { train: 0.8, validation: 0.3, holdout: 0.1 }), /must sum to 1/);
  assert.throws(() => splitRecords(records, { train: 1, validation: 0, holdout: 0 }), /holdout split is empty/);
});

test('dataset assignments preserve pre-registered train, validation and holdout', () => {
  const names = ['script-train', 'script-validation', 'script-holdout'];
  const records = names.flatMap((name, group) => Array.from({ length: 12 }, (_, i) => ({
    id: `${group}-${i}`, text: `distinct story ${group} item ${i}`,
    prompt: `prompt ${group} ${i}`, completion: `answer ${group} ${i}`,
    source: { dataset: name },
  })));
  const config = { datasetAssignments: Object.fromEntries(names.map((name, i) => [name, ['train', 'validation', 'holdout'][i]])) };
  const splits = splitRecords(records, config);
  assert.deepEqual(Object.fromEntries(Object.entries(splits).map(([key, values]) => [key, values.length])), {
    train: 12, validation: 12, holdout: 12,
  });
  assert.ok(splits.holdout.every((row) => row.source.dataset === 'script-holdout'));
  assert.throws(() => splitRecords([...records, { id: 'unmapped', text: 'unmapped', source: { dataset: 'other' } }], config), /no valid assignment/);
});

// Records built from a shared template are near-duplicates of each other by
// construction, which would make the leakage gate fire on the baseline corpus.
// These fixtures sample a vocabulary instead, so any hit is the injected one.
const LEAKAGE_VOCAB = (
  'billing outage access hiring legal shipping refund invoice incident review renewal audit '
  + 'critical urgent routine later monday friday weekend emea apac latam nordics benelux '
  + 'finance platform security people compliance logistics escalation handoff approval '
  + 'rejection backlog roadmap vendor contract'
).split(' ');

function variedRecord(i) {
  const rng = new Rng(`leakage-doc-${i}`);
  return { id: `r${i}`, text: Array.from({ length: 14 }, () => rng.pick(LEAKAGE_VOCAB)).join(' ') };
}

// negative:1 from the acceptance plan.
test('leakage check blocks a holdout sample injected into the training set', () => {
  const records = Array.from({ length: 120 }, (_, i) => variedRecord(i));
  const splits = splitRecords(records);
  assert.ok(checkLeakage(splits).ok);

  splits.train.push({ ...splits.holdout[0], split: 'train' });
  const result = checkLeakage(splits);
  assert.equal(result.ok, false);
  assert.equal(result.counts.exact, 1);
  assert.equal(result.counts.near, 1, 'the same text is also a near-duplicate hit');
  assert.throws(() => assertNoLeakage(splits), /leakage check failed/);
  try {
    assertNoLeakage(splits);
    assert.fail('expected assertNoLeakage to throw');
  } catch (err) {
    assert.deepEqual(err.leakage.exact[0].splits, ['holdout', 'train']);
  }
});

test('leakage check catches a lightly edited holdout record', () => {
  const records = Array.from({ length: 120 }, (_, i) => variedRecord(i));
  const splits = splitRecords(records);
  assert.ok(checkLeakage(splits).ok, 'the fixture corpus must start clean');
  const target = splits.holdout[0];
  splits.train.push({ ...target, id: 'paraphrase', text: `${target.text} followup`, split: 'train' });
  const result = checkLeakage(splits);
  assert.equal(result.counts.exact, 0, 'the canonical keys differ');
  assert.equal(result.counts.near, 1, 'the near-duplicate path must still catch it');
  assert.ok(result.near[0].similarity >= 0.85);
});

test('the full pipeline cleans, redacts, dedupes, splits and reports without leaking values', async () => {
  const dir = await tmp('tmf-pipeline-');
  const corpus = await writeSyntheticCorpus(path.join(dir, 'corpus'));
  const out = path.join(dir, 'data');
  const { report, splits } = await runDataPipeline({ datasets: corpus.datasets, outDir: out, baseDir: dir });

  assert.ok(report.stageCounts.ingested > 240);
  assert.ok(report.stageCounts.exactDuplicates >= 6, 'injected exact duplicates must be removed');
  assert.ok(report.stageCounts.nearDuplicates >= 1, 'the injected typo re-submission must be removed');
  assert.ok(report.stageCounts.qualityDropped >= 3, 'junk records must be removed');
  assert.ok(report.stageCounts.piiRedacted >= 4, 'records with sensitive data must be redacted');
  assert.equal(report.stageCounts.piiDropped, 0);
  assert.ok(report.leakage.ok);
  assert.deepEqual(report.provenance.licenses, ['CC0-1.0']);
  assert.equal(report.sensitiveInfo.policy, 'redact');
  assert.ok(report.sensitiveInfo.byKind.email >= 1 && report.sensitiveInfo.byKind.credential >= 1);

  // The report is an audit artifact: it must not restate the sensitive values.
  const serialized = JSON.stringify(report);
  for (const secret of SECRETS) assert.ok(!serialized.includes(secret), `report leaked ${secret}`);
  const written = await readJsonl(path.join(out, 'train.jsonl'));
  const trainSerialized = JSON.stringify(written);
  for (const secret of SECRETS) assert.ok(!trainSerialized.includes(secret), `training data leaked ${secret}`);
  assert.equal(written.length, splits.train.length);
  assert.ok(splits.holdout.length > 10);
});

test('the pipeline is idempotent for a fixed corpus and seed', async () => {
  const dir = await tmp('tmf-idem-');
  const corpus = await writeSyntheticCorpus(path.join(dir, 'corpus'), { taskCount: 80, pretrainCount: 20 });
  const first = await runDataPipeline({ datasets: corpus.datasets, baseDir: dir });
  const second = await runDataPipeline({ datasets: corpus.datasets, baseDir: dir });
  assert.equal(first.report.configDigest, second.report.configDigest);
  assert.deepEqual(first.report.stageCounts, second.report.stageCounts);
  assert.deepEqual(first.report.splits, second.report.splits);
  assert.deepEqual(first.splits.holdout.map((r) => r.id), second.splits.holdout.map((r) => r.id));
});

test('the pipeline aborts when the policy is fail and the corpus carries secrets', async () => {
  const dir = await tmp('tmf-fail-');
  await mkdir(path.join(dir, 'raw'), { recursive: true });
  await writeFile(
    path.join(dir, 'raw', 'leaky.jsonl'),
    '{"text":"the deployment key is sk-abcdefghijklmnopqrstuvwx and must be rotated"}\n',
  );
  await assert.rejects(
    runDataPipeline({
      datasets: [{ name: 'leaky', path: path.join(dir, 'raw'), license: 'CC0-1.0' }],
      baseDir: dir,
      config: { pii: { policy: 'fail' }, split: { requireNonEmptyHoldout: false } },
    }),
    /pii policy "fail"/,
  );
});

test('the pipeline removes duplicates before splitting and the gate still catches contamination', async () => {
  const dir = await tmp('tmf-leak-');
  await mkdir(path.join(dir, 'raw'), { recursive: true });
  const rows = Array.from({ length: 60 }, (_, i) => JSON.stringify({ text: `incident ${i} about the platform outage review and handoff` }));
  rows.push(rows[0]);
  await writeFile(path.join(dir, 'raw', 'a.jsonl'), `${rows.join('\n')}\n`);
  const datasets = [{ name: 'raw', path: path.join(dir, 'raw'), license: 'CC0-1.0' }];

  const deduped = await runDataPipeline({ datasets, baseDir: dir });
  assert.ok(deduped.report.leakage.ok, 'with dedupe on, the duplicate is removed before splitting');
  assert.ok(deduped.report.stageCounts.exactDuplicates >= 1);

  // Force the contamination the gate exists for.
  deduped.splits.train.push({ ...deduped.splits.holdout[0], split: 'train' });
  assert.throws(() => assertNoLeakage(deduped.splits), /leakage check failed/);
});

test('synthetic task records are internally consistent with the routing rules', () => {
  const records = generateTaskRecords(60, 'seed-a');
  assert.equal(records.length, 60);
  assert.equal(new Set(records.map((r) => r.meta.subject)).size, 60, 'subjects must be unique');
  for (const record of records) {
    const expected = expectedRouting(record.meta.subject);
    assert.equal(expected.queue, record.meta.queue);
    assert.equal(expected.priority, record.meta.priority);
    assert.equal(record.completion, `\nqueue=${expected.queue}\npriority=${expected.priority}\n`);
  }
  assert.deepEqual(generateTaskRecords(5, 'seed-a'), generateTaskRecords(5, 'seed-a'));
  assert.notDeepEqual(generateTaskRecords(5, 'seed-a'), generateTaskRecords(5, 'seed-b'));
});

// train-ws-full put 2 of ~40 supervised pairs into a 25-record holdout, so
// exact match was estimated on two samples. Stratified allocation is what
// stops that: the pair population gets its own proportional split.
test('holdout keeps its proportional share of supervised pairs', () => {
  const pairs = Array.from({ length: 40 }, (_, i) => ({
    id: `p${i}`,
    text: `instruction ${i}\nqueue=q${i} priority=p${i % 3}`,
    prompt: `instruction ${i}`,
    completion: `queue=q${i} priority=p${i % 3}`,
  }));
  const docs = Array.from({ length: 160 }, (_, i) => ({ id: `d${i}`, text: `handbook section ${i} covering escalation policy ${i}` }));
  const records = [...pairs, ...docs];

  // This case is about the allocation arithmetic, not sample sufficiency: 40
  // pairs cannot fill the default scorable-holdout minimum, so opt out of it.
  const quotaOnly = { minScorableHoldout: 0 };
  const stratified = splitRecords(records, quotaOnly);
  const heldOutPairs = stratified.holdout.filter((r) => r.stratum === 'pair');
  assert.equal(heldOutPairs.length, 6, '15% of 40 pairs must be held out');
  assert.equal(stratified.holdout.length, 6 + 24);
  assert.equal(stratified.train.filter((r) => r.stratum === 'pair').length, 28);

  // Still order-independent and still deterministic.
  const reversed = splitRecords(records.slice().reverse(), quotaOnly);
  const ids = (s) => s.map((r) => r.id).sort();
  for (const name of ['train', 'validation', 'holdout']) {
    assert.deepEqual(ids(reversed[name]), ids(stratified[name]), `${name} must not depend on input order`);
  }

  // The unstratified path is the behaviour that produced the defect: the pair
  // count in holdout is binomial there, so it is only an escape hatch and
  // carries no quota guarantee.
  const unstratified = splitRecords(records, { ...quotaOnly, stratify: false });
  assert.ok(unstratified.holdout.length > 0);
});

test('a holdout with no supervised pair is rejected rather than scored', () => {
  const records = [
    { id: 'p0', text: 'instruction 0\nqueue=q0', prompt: 'instruction 0', completion: 'queue=q0' },
    ...Array.from({ length: 40 }, (_, i) => ({ id: `d${i}`, text: `handbook section ${i} about refunds ${i}` })),
  ];
  // One pair cannot fill a 15% quota, so the pair stratum has no holdout slot.
  assert.throws(() => splitRecords(records), /no supervised pair/);
  assert.ok(splitRecords(records, { requireScorableHoldout: false }).holdout.length > 0);
});

test('a holdout too small to score exact match is rejected at data time', () => {
  const docs = Array.from({ length: 160 }, (_, i) => ({ id: `d${i}`, text: `handbook section ${i} covering escalation policy ${i}` }));
  const pairsFor = (n) => Array.from({ length: n }, (_, i) => ({
    id: `p${i}`,
    text: `instruction ${i}\nqueue=q${i} priority=p${i % 3}`,
    prompt: `instruction ${i}`,
    completion: `queue=q${i} priority=p${i % 3}`,
  }));

  // 48 pairs is what the CLI first run used to generate: a 15% share is 7
  // scorable records, below eval's minimum, so the job must fail here rather
  // than train and then report an exact-match rate over 7 samples.
  assert.equal(MIN_SCORABLE_HOLDOUT, 10, 'the data minimum must track eval MIN_EXACT_MATCH_SAMPLES');
  assert.throws(
    () => splitRecords([...pairsFor(48), ...docs]),
    (err) => /below the minimum 10/.test(err.message) && /at least 67 supervised pair/.test(err.message),
  );

  // Enough pairs and the same config passes untouched, with the quota intact.
  const ok = splitRecords([...pairsFor(80), ...docs]);
  assert.equal(ok.holdout.filter((r) => r.stratum === 'pair').length, 12);

  // The minimum is a guard, not a silent clamp: an explicit opt-out still
  // yields the same assignment, it just accepts the smaller sample.
  const accepted = splitRecords([...pairsFor(48), ...docs], { minScorableHoldout: 0 });
  assert.equal(accepted.holdout.filter((r) => r.stratum === 'pair').length, 7);
});
