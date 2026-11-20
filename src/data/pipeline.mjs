// Data pipeline: ingest -> clean -> sensitive-info policy -> quality -> dedupe
// -> provenance -> split -> leakage check -> report.
//
// Two rules shape the output. First, the report records counts and reasons but
// never the offending text, so an audit trail can be shared without re-leaking
// the data it is auditing. Second, the stage order puts redaction before
// dedupe and splitting, so a secret cannot survive by hiding in a duplicate.

import path from 'node:path';
import { sha256Json } from '../core/hash.mjs';
import { writeJsonl, writeJson, ensureDir } from '../core/fsx.mjs';
import { ingestAll } from './ingest.mjs';
import { cleanText, qualityReport, canonicalKey, DEFAULT_CLEAN_CONFIG, DEFAULT_QUALITY_CONFIG } from './clean.mjs';
import { applyPiiPolicy } from './pii.mjs';
import { NearDuplicateIndex, DEFAULT_DEDUPE_CONFIG, exactKey } from './dedupe.mjs';
import { summarizeProvenance, assertTrainableClassification, DEFAULT_TRAINABLE_CLASSIFICATIONS } from './provenance.mjs';
import { splitRecords, checkLeakage, DEFAULT_SPLIT_CONFIG, SPLIT_NAMES } from './split.mjs';

export const DEFAULT_PIPELINE_CONFIG = {
  clean: DEFAULT_CLEAN_CONFIG,
  quality: DEFAULT_QUALITY_CONFIG,
  dedupe: DEFAULT_DEDUPE_CONFIG,
  split: DEFAULT_SPLIT_CONFIG,
  pii: { policy: 'redact', blockSeverity: 'critical' },
  allowedClassifications: DEFAULT_TRAINABLE_CLASSIFICATIONS,
  failOnLeakage: true,
  maxRecords: null,
};

function bump(map, key) {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function mapToObject(map) {
  return Object.fromEntries([...map.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)));
}

export async function runDataPipeline({ datasets, outDir, baseDir = process.cwd(), config = {}, onProgress } = {}) {
  const cfg = {
    ...DEFAULT_PIPELINE_CONFIG,
    ...config,
    clean: { ...DEFAULT_CLEAN_CONFIG, ...(config.clean ?? {}) },
    quality: { ...DEFAULT_QUALITY_CONFIG, ...(config.quality ?? {}) },
    dedupe: { ...DEFAULT_DEDUPE_CONFIG, ...(config.dedupe ?? {}) },
    split: { ...DEFAULT_SPLIT_CONFIG, ...(config.split ?? {}) },
    pii: { ...DEFAULT_PIPELINE_CONFIG.pii, ...(config.pii ?? {}) },
  };

  const ingested = await ingestAll(datasets, { baseDir });
  onProgress?.({ stage: 'ingest', records: ingested.records.length, errors: ingested.errors.length });

  const stageCounts = { ingested: ingested.records.length, cleaned: 0, piiRedacted: 0, piiDropped: 0, qualityDropped: 0, exactDuplicates: 0, nearDuplicates: 0, accepted: 0 };
  const dropReasons = new Map();
  const piiKinds = new Map();
  const piiSeverities = new Map();
  const cleanOps = new Map();

  const accepted = [];
  const seenExact = new Map();
  const nearIndex = cfg.dedupe.nearDuplicates ? new NearDuplicateIndex(cfg.dedupe) : null;

  for (const record of ingested.records) {
    if (cfg.maxRecords && accepted.length >= cfg.maxRecords) break;

    const cleanedPrompt = record.prompt === null ? null : cleanText(record.prompt, cfg.clean);
    const cleanedCompletion = record.completion === null ? null : cleanText(record.completion, { ...cfg.clean, trim: false });
    const cleanedText = cleanText(record.text, cfg.clean);
    for (const op of new Set([...(cleanedPrompt?.applied ?? []), ...(cleanedCompletion?.applied ?? []), ...cleanedText.applied])) bump(cleanOps, op);
    stageCounts.cleaned += 1;

    // Sensitive-info policy runs on the full record text so a secret split
    // across prompt and completion is still caught.
    let pii;
    try {
      pii = applyPiiPolicy(cleanedText.text, cfg.pii);
    } catch (err) {
      // policy: 'fail' aborts the job; surface which record without the content.
      err.recordId = record.id;
      err.source = record.source;
      throw err;
    }
    for (const finding of pii.findings) {
      bump(piiKinds, finding.kind);
      bump(piiSeverities, finding.severity);
    }
    if (pii.action === 'drop') {
      stageCounts.piiDropped += 1;
      bump(dropReasons, `pii_${pii.worst}`);
      continue;
    }

    let prompt = cleanedPrompt?.text ?? null;
    let completion = cleanedCompletion?.text ?? null;
    let text = cleanedText.text;
    if (pii.action === 'redact' && pii.findings.length > 0) {
      stageCounts.piiRedacted += 1;
      text = pii.text;
      if (prompt !== null) prompt = applyPiiPolicy(prompt, cfg.pii).text;
      if (completion !== null) completion = applyPiiPolicy(completion, cfg.pii).text;
    }

    const quality = qualityReport(text, cfg.quality);
    if (!quality.ok) {
      stageCounts.qualityDropped += 1;
      for (const reason of quality.reasons) bump(dropReasons, `quality_${reason.split('(')[0]}`);
      continue;
    }

    const key = exactKey(text);
    if (seenExact.has(key)) {
      stageCounts.exactDuplicates += 1;
      bump(dropReasons, 'exact_duplicate');
      continue;
    }

    if (nearIndex) {
      const { signature, match } = nearIndex.query(text);
      if (match) {
        stageCounts.nearDuplicates += 1;
        bump(dropReasons, 'near_duplicate');
        continue;
      }
      nearIndex.add(record.id, text, signature);
    }

    seenExact.set(key, record.id);
    accepted.push({
      id: record.id,
      prompt,
      completion,
      text,
      source: record.source,
      quality: { score: quality.score, chars: quality.metrics.chars },
      piiRedacted: pii.action === 'redact' && pii.findings.length > 0,
      canonicalPreviewHash: sha256Json({ key: canonicalKey(text) }).slice(0, 16),
    });
  }
  stageCounts.accepted = accepted.length;
  onProgress?.({ stage: 'filter', accepted: accepted.length });

  assertTrainableClassification(accepted, cfg.allowedClassifications);

  const splits = splitRecords(accepted, cfg.split);
  const leakage = checkLeakage(splits, { dedupe: cfg.dedupe });
  onProgress?.({ stage: 'split', sizes: Object.fromEntries(SPLIT_NAMES.map((n) => [n, splits[n].length])) });

  if (cfg.failOnLeakage && !leakage.ok) {
    const err = new Error(
      `data pipeline aborted: leakage check failed (${leakage.counts.exact} exact, ${leakage.counts.near} near-duplicate across splits)`,
    );
    err.leakage = leakage;
    throw err;
  }

  const provenance = summarizeProvenance(accepted);
  const report = {
    version: 1,
    generatedAt: new Date().toISOString(),
    configDigest: sha256Json(cfg),
    datasets: ingested.datasets,
    ingestErrors: ingested.errors.length,
    ingestErrorSamples: ingested.errors.slice(0, 5).map((e) => ({ file: e.file, line: e.line, reason: e.reason })),
    stageCounts,
    dropReasons: mapToObject(dropReasons),
    cleanOperations: mapToObject(cleanOps),
    // Counts only: the matched values are never written to disk.
    sensitiveInfo: { byKind: mapToObject(piiKinds), bySeverity: mapToObject(piiSeverities), policy: cfg.pii.policy },
    provenance,
    splits: Object.fromEntries(SPLIT_NAMES.map((n) => [n, splits[n].length])),
    splitConfig: cfg.split,
    leakage,
  };

  if (outDir) {
    await ensureDir(outDir);
    for (const name of SPLIT_NAMES) {
      await writeJsonl(path.join(outDir, `${name}.jsonl`), splits[name]);
    }
    await writeJson(path.join(outDir, 'data-report.json'), report);
  }

  return { report, splits, provenance, accepted, config: cfg };
}
