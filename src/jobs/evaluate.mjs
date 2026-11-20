// The evaluation job.
//
// A re-runnable unit of work that turns a checkpoint pair plus held-out data
// into a gate verdict and a lineage-bearing report. It is deliberately the
// only place allowed to decide whether a run may be described as an
// improvement: `verdict.improvementClaimAllowed` is false unless the gates
// actually passed, and `assertDeliverable` is what export paths call so a
// failed gate is terminal rather than advisory.

import path from 'node:path';
import { evaluateGates, DEFAULT_GATES } from '../eval/gates.mjs';
import {
  EVAL_REPORT_FORMAT, assertHoldoutIndependence, evaluateSplitSet, assembleMetrics,
} from '../eval/evaluator.mjs';
import { sha256File, sha256Json } from '../core/hash.mjs';
import { writeJson, exists } from '../core/fsx.mjs';

// A report handed in from outside (`tmf export --report`) is only usable as a
// gate record if it still hashes to its own `reportDigest`: otherwise the
// bundle would claim lineage for numbers nobody evaluated.
export function assertReportIntegrity(report, { jobId = null } = {}) {
  if (!report || typeof report !== 'object') throw new Error('eval report must be an object');
  if (report.format !== EVAL_REPORT_FORMAT) {
    throw new Error(`eval report format mismatch: expected ${EVAL_REPORT_FORMAT}, got ${report.format ?? 'none'}`);
  }
  if (!report.reportDigest) throw new Error('eval report is missing reportDigest');
  // Mirror how the digest was produced: `reportDigest` stays as an undefined
  // key (which canonicalizes to null), while `reportFile` — attached only
  // in-memory after the write — must be absent, not undefined.
  const canonical = { ...report, reportDigest: undefined };
  delete canonical.reportFile;
  const recomputed = sha256Json(canonical);
  if (recomputed !== report.reportDigest) {
    throw new Error(`eval report digest mismatch: file claims ${report.reportDigest}, contents hash to ${recomputed}`);
  }
  const reportJob = report.lineage?.jobId ?? null;
  if (jobId && reportJob && reportJob !== jobId) {
    throw new Error(`eval report belongs to job ${reportJob}, not ${jobId}`);
  }
  return report;
}

async function checkpointRef(file) {
  if (!file) return null;
  if (!(await exists(file))) throw new Error(`checkpoint not found: ${file}`);
  return { file, sha256: await sha256File(file) };
}

export async function runEvaluationJob({
  backend,
  baseCheckpoint,
  adapterCheckpoint = null,
  tunedCheckpoint = null,
  encoded,
  records = {},
  gates = DEFAULT_GATES,
  lineage = {},
  reportFile = null,
  maxNewTokens = 48,
  jobId = null,
  clock = () => new Date().toISOString(),
} = {}) {
  if (!backend) throw new Error('runEvaluationJob needs a backend');
  if (!baseCheckpoint) throw new Error('runEvaluationJob needs a base checkpoint');
  const startedAt = clock();

  const independence = assertHoldoutIndependence({
    holdout: records.holdout ?? [],
    train: records.train ?? [],
    validation: records.validation ?? [],
  });

  const baseRef = await checkpointRef(baseCheckpoint);
  const adapterRef = await checkpointRef(adapterCheckpoint);
  const tunedRef = await checkpointRef(tunedCheckpoint);

  // Baseline is the base checkpoint with no adapter attached. For a LoRA run
  // the tuned side is that same base with the adapter layered on; for a
  // full-parameter run (pretrain / full-finetune) the trainer writes a whole
  // tuned checkpoint instead, and that file is the tuned side. Either way both
  // sides are loaded from disk, which is what makes the comparison independent
  // of trainer state -- and without the full-checkpoint path those modes could
  // never produce gates, so export stayed permanently blocked for them.
  const baselineHandle = await backend.loadModel({ file: baseCheckpoint });
  const baselineFingerprint = backend.fingerprint(baselineHandle);
  const baseline = await evaluateSplitSet(backend, baselineHandle, { encoded, records, maxNewTokens });

  let tuned = null;
  let tunedFingerprint = null;
  if (adapterCheckpoint || tunedCheckpoint) {
    const tunedHandle = adapterCheckpoint
      ? await backend.loadModel({ file: baseCheckpoint, attachAdapter: adapterCheckpoint })
      : await backend.loadModel({ file: tunedCheckpoint });
    tunedFingerprint = backend.fingerprint(tunedHandle);
    tuned = await evaluateSplitSet(backend, tunedHandle, { encoded, records, maxNewTokens });
  }

  const metrics = assembleMetrics({ baseline, tuned });
  const verdictGates = tuned ? evaluateGates(metrics, gates) : null;
  const status = !tuned ? 'baseline_only' : (verdictGates.passed ? 'passed' : 'failed');

  const report = {
    format: EVAL_REPORT_FORMAT,
    jobId: jobId ?? `eval-${startedAt}`,
    startedAt,
    finishedAt: clock(),
    backend: { id: backend.id, label: backend.label ?? null },
    inputs: {
      baseCheckpoint: baseRef,
      adapterCheckpoint: adapterRef,
      tunedCheckpoint: tunedRef,
      splits: {
        holdout: encoded.holdout?.length ?? 0,
        validation: encoded.validation?.length ?? 0,
        trainWithheld: records.train?.length ?? 0,
      },
    },
    independence,
    metrics,
    gates: verdictGates ?? { passed: false, results: [], failed: [], summary: 'no tuned model to gate' },
    verdict: {
      status,
      improvementClaimAllowed: status === 'passed',
      deliveryEligible: status === 'passed',
      summary: status === 'passed'
        ? `improvement confirmed on an independent holdout (${verdictGates.summary})`
        : status === 'failed'
          ? `gates failed, no improvement may be claimed (${verdictGates.summary}): ${verdictGates.failed.map((f) => f.reason).join('; ')}`
          : 'baseline-only evaluation: no tuned checkpoint was supplied, so no improvement may be claimed',
    },
    lineage: {
      ...lineage,
      // The builtin fingerprint digests the *base* weights, so for an adapter
      // run both handles share it -- that equality is the proof the adapter is
      // bound to this base checkpoint. Adapter bytes are pinned by
      // inputs.adapterCheckpoint.sha256.
      baseFingerprint: baselineFingerprint,
      tunedFingerprint,
      // Only an adapter run can be "bound to base". A full-parameter tuned
      // checkpoint has no adapter, so reporting false there would read as a
      // broken binding instead of "not applicable".
      adapterBoundToBase: adapterCheckpoint ? tunedFingerprint === baselineFingerprint : null,
      gateDefinitions: gates.map((g) => ({ id: g.id, metric: g.metric, comparator: g.comparator, threshold: g.threshold })),
      gateDefinitionsDigest: sha256Json(gates),
    },
  };
  report.reportDigest = sha256Json({ ...report, reportDigest: undefined });

  if (reportFile) {
    await writeJson(reportFile, report);
    report.reportFile = path.resolve(reportFile);
  }
  return report;
}

// Delivery/export gate. Callers that want the non-throwing form read
// `report.verdict.deliveryEligible` instead.
export function assertDeliverable(report) {
  if (report?.verdict?.deliveryEligible) return report;
  const err = new Error(`evaluation report is not deliverable: ${report?.verdict?.summary ?? 'missing verdict'}`);
  err.code = 'EVAL_NOT_DELIVERABLE';
  err.report = report;
  throw err;
}
