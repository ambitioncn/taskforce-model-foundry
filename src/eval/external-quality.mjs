// Bind an independently scored, predeclared quality suite to a trained job.
// The scorer is outside Foundry; Foundry verifies its evidence and gates.
import path from 'node:path';
import { readJson } from '../core/fsx.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { evaluateGates, readMetric } from './gates.mjs';
import { ExportBlockedError } from '../export/package.mjs';
import { SCREENPLAY_PROFILE, screenplayGatesMatch } from './screenplay-profile.mjs';

export const EXTERNAL_QUALITY_FORMAT = 'tmf.external-quality/v1';
export const externalQualityContractDigest = (contract) => sha256Json(contract);

// Convenience for an external scorer. This is an integrity checksum, not a
// signature or proof that the scorer's measurements are truthful.
export function sealExternalQualityEvidence(report) {
  return { ...report, reportDigest: sha256Json({ ...report, reportDigest: undefined }) };
}

function blocked(message, failed = []) {
  throw new ExportBlockedError(`export blocked: ${message}`, { status: 'external_quality', failed });
}

export async function verifyExternalQuality({ contract, evidence, artifacts, evalReport, baseDir }) {
  if (!evidence) blocked('predeclared external quality evidence is missing');
  if (typeof evidence !== 'string') blocked('external quality evidence must be a report file path');
  const file = path.resolve(baseDir, evidence);
  let report;
  try { report = await readJson(file); } catch (err) { blocked(`external quality report is unreadable: ${err.message}`); }
  if (!report || report.format !== EXTERNAL_QUALITY_FORMAT) blocked('external quality report format mismatch');
  if (contract.profile === SCREENPLAY_PROFILE && (report.profile !== SCREENPLAY_PROFILE || !screenplayGatesMatch(contract.gates))) {
    blocked('screenplay report or frozen gate set does not match screenplay-zh-v1');
  }
  const digest = sha256Json({ ...report, reportDigest: undefined });
  if (report.reportDigest !== digest) blocked('external quality report digest mismatch');
  if (report.jobId !== artifacts.jobId) blocked('external quality report belongs to another job');
  if (report.suiteSha256 !== contract.suiteSha256 || report.scorerSha256 !== contract.scorerSha256) {
    blocked('external quality suite/scorer differs from the predeclared contract');
  }
  if (report.contractDigest !== sha256Json(contract)) blocked('external quality contract digest mismatch');
  if (report.evalReportDigest !== evalReport.reportDigest) blocked('external quality report is bound to a different technical evaluation');
  if (!evalReport.lineage?.tunedFingerprint || report.tunedFingerprint !== evalReport.lineage.tunedFingerprint) {
    blocked('external quality report is bound to a different tuned model fingerprint');
  }
  const checkpoint = artifacts.adapterCheckpoint ?? artifacts.tunedCheckpoint;
  if (!checkpoint || report.checkpointSha256 !== await sha256File(checkpoint)) blocked('external quality report is bound to a different trained checkpoint');
  for (const gate of contract.gates) {
    if (!Number.isFinite(readMetric(report.metrics, gate.metric))) blocked(`external quality metric ${gate.metric} is not a finite number`);
    if (gate.minSamples && !Number.isInteger(readMetric(report.metrics, gate.sampleMetric))) {
      blocked(`external quality sample count ${gate.sampleMetric} is not an integer`);
    }
  }
  const gates = evaluateGates(report.metrics ?? {}, contract.gates);
  if (!gates.passed) blocked(`external quality gates failed (${gates.summary})`, gates.failed.map((gate) => gate.id));
  return { file, fileSha256: await sha256File(file), report, gates, contractDigest: sha256Json(contract) };
}
