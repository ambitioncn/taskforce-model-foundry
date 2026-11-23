import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { validateConfig } from '../src/config.mjs';
import { sha256File, sha256Json } from '../src/core/hash.mjs';
import { EXTERNAL_QUALITY_FORMAT, verifyExternalQuality } from '../src/eval/external-quality.mjs';
import { qualityContext } from '../src/api.mjs';

const digest = 'a'.repeat(64);
const contract = {
  suiteSha256: digest, scorerSha256: 'b'.repeat(64),
  gates: [{ id: 'domain_quality', metric: 'quality.overall', comparator: 'gte', threshold: 0.7 }],
};

test('external quality contract is validated before training', () => {
  const raw = { eval: { externalQuality: contract } };
  const valid = validateConfig(raw);
  assert.equal(valid.ok, true);
  assert.deepEqual(valid.config.eval.externalQuality, contract);
  const invalid = validateConfig({ eval: { externalQuality: { ...contract, suiteSha256: 'bad', gates: [] } } });
  assert.equal(invalid.ok, false);
  assert.ok(invalid.errors.some((e) => e.at === 'eval.externalQuality.suiteSha256'));
  const typed = { ...contract, profile: 'customer-support-v1', references: { baseline: 'c'.repeat(64) } };
  assert.deepEqual(validateConfig({ eval: { externalQuality: typed } }).config.eval.externalQuality, typed);
  const invalidReference = validateConfig({ eval: { externalQuality: { ...typed, references: { baseline: 'bad' } } } });
  assert.ok(invalidReference.errors.some((e) => e.at === 'eval.externalQuality.references.baseline'));
});

test('external evidence must match frozen suite, scorer, job, checkpoint and technical report', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'tmf-quality-'));
  const checkpoint = path.join(dir, 'adapter.json');
  const evidenceFile = path.join(dir, 'quality.json');
  await writeFile(checkpoint, 'frozen adapter');
  const artifacts = { jobId: 'job-a', adapterCheckpoint: checkpoint };
  const evalReport = { reportDigest: 'c'.repeat(64), lineage: { tunedFingerprint: 'e'.repeat(64) } };
  const base = {
    format: EXTERNAL_QUALITY_FORMAT,
    jobId: artifacts.jobId,
    checkpointSha256: await sha256File(checkpoint),
    evalReportDigest: evalReport.reportDigest,
    tunedFingerprint: evalReport.lineage.tunedFingerprint,
    contractDigest: sha256Json(contract),
    suiteSha256: contract.suiteSha256,
    scorerSha256: contract.scorerSha256,
    metrics: { quality: { overall: 0.75 } },
  };
  const writeEvidence = async (change = {}) => {
    const report = { ...base, ...change };
    report.reportDigest = sha256Json({ ...report, reportDigest: undefined });
    await writeFile(evidenceFile, JSON.stringify(report));
  };
  const verify = () => verifyExternalQuality({ contract, evidence: evidenceFile, artifacts, evalReport, baseDir: dir });
  await assert.rejects(verify, { code: 'EXPORT_BLOCKED' });
  await writeEvidence();
  assert.equal((await verify()).gates.passed, true);
  const typedContract = { ...contract, profile: 'customer-support-v1' };
  await writeEvidence({ profile: 'wrong-profile', contractDigest: sha256Json(typedContract) });
  await assert.rejects(() => verifyExternalQuality({ contract: typedContract, evidence: evidenceFile,
    artifacts, evalReport, baseDir: dir }), /profile differs/);
  await writeEvidence();
  await writeEvidence({ metrics: { quality: { overall: 0.3 } } });
  await assert.rejects(verify, /gates failed/);
  await writeEvidence({ checkpointSha256: digest });
  await assert.rejects(verify, /different trained checkpoint/);
  await writeEvidence({ suiteSha256: 'd'.repeat(64) });
  await assert.rejects(verify, /suite\/scorer differs/);
  await writeEvidence({ evalReportDigest: digest });
  await assert.rejects(verify, /different technical evaluation/);
  await writeEvidence({ tunedFingerprint: digest });
  await assert.rejects(verify, /different tuned model fingerprint/);
  await writeEvidence();
  const tampered = JSON.parse(await (await import('node:fs/promises')).readFile(evidenceFile, 'utf8'));
  tampered.metrics.quality.overall = 0.01;
  await writeFile(evidenceFile, JSON.stringify(tampered));
  await assert.rejects(verify, /digest mismatch/);
});

test('quality context reloads the frozen contract and refuses failed technical gates', async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'tmf-quality-context-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const jobId = 'quality-fixture';
  const runRoot = path.join(dir, 'runs');
  const configFile = path.join(dir, 'tmf.config.json');
  const checkpoint = path.join(dir, 'adapter.json');
  await mkdir(path.join(runRoot, jobId), { recursive: true });
  await writeFile(checkpoint, 'adapter bytes');
  await writeFile(configFile, JSON.stringify({ format: 'tmf-config-v1', project: 'quality-fixture', runRoot,
    eval: { externalQuality: contract } }));
  const manifest = { jobId, data: { dataDir: path.join(dir, 'data') }, base: { file: path.join(dir, 'base.json') },
    train: { kind: 'adapter', checkpoint }, externalQuality: contract };
  await writeFile(path.join(runRoot, jobId, 'state.json'), JSON.stringify({ jobId, state: 'succeeded',
    completedSteps: [{ name: 'finalize', output: manifest }] }));
  const technical = { format: 'tmf-eval-report-v1', lineage: { jobId, tunedFingerprint: 'd'.repeat(64),
    externalQualityContractDigest: sha256Json(contract) }, verdict: { status: 'failed', deliveryEligible: false } };
  const seal = (report) => ({ ...report, reportDigest: sha256Json({ ...report, reportDigest: undefined }) });
  await assert.rejects(() => qualityContext({ configFile, jobId, evalReport: seal(technical) }), /passed technical/);
  const passed = seal({ ...technical, verdict: { status: 'passed', deliveryEligible: true } });
  const context = await qualityContext({ configFile, jobId, evalReport: passed });
  assert.deepEqual(context.contract, contract);
  assert.equal(context.checkpointSha256, await sha256File(checkpoint));
  assert.equal(context.evalReport.reportDigest, passed.reportDigest);
});
