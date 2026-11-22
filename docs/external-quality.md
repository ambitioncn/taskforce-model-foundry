# External quality gate for screenplay models

Foundry's independent loss/strict-match report is a **technical** gate, not a
Chinese screenplay quality judgement. A project may require separately scored
quality evidence before export. This mechanism does not run the scorer or
claim that a checksum proves the scores are truthful.

1. **Before starting a new training job**, add `eval.externalQuality` to
   `tmf.config.json`. Freeze the complete suite and scorer files; put their
   SHA-256 digests in `suiteSha256` and `scorerSha256`. Declare nonempty gates
   using the same `id`, `metric`, `comparator`, `threshold`, optional
   `sampleMetric`/`minSamples` syntax as `eval.gates`. The training manifest
   and job input digest pin this contract. Do not retrofit it onto an old job.
2. Evaluate the trained checkpoint with `tmf evaluate`. For the 120-question
   Chinese screenplay suite, use the already frozen
   `screenplay_zh_eval_v1.json` (SHA-256
   `1344ab895d1e35c5b5603c73cb3a832d924b9a6d1159accd40c95def511216fd`)
   and the predeclared blind scorer. The original screenplay plan requires
   hard-constraint lift vs DAPT ≥5 percentage points, anonymous A/B win rate
   ≥55%, scene completion ≥95%, and early-EOS rate ≤5%, alongside category
   non-regression. Translate **all** required criteria into contract gates
   before looking at candidate results. No default thresholds are silently
   supplied by Foundry.
3. The external scorer writes a JSON report with:

   ```json
   {
     "format": "tmf.external-quality/v1",
     "jobId": "new-training-job-id",
     "suiteSha256": "<predeclared-suite-sha256>",
     "scorerSha256": "<predeclared-scorer-sha256>",
     "contractDigest": "<sha256Json(eval.externalQuality)>",
     "checkpointSha256": "<sha256 of trained adapter or tuned checkpoint file>",
     "evalReportDigest": "<technical eval reportDigest>",
     "tunedFingerprint": "<technical eval lineage.tunedFingerprint>",
     "metrics": { "quality": { "overall": 0.0 } },
     "reportDigest": "<canonical digest of this report without reportDigest>"
   }
   ```

   The scorer may import `externalQuality.sealExternalQualityEvidence` from
   `taskforce-model-foundry` to compute the canonical `reportDigest`; the
   `contractDigest` can be computed with
   `externalQuality.externalQualityContractDigest(contract)`. These
   digests detect edits and bind the evidence, but are **not signatures**.
   Preserve the raw per-question outputs, blinded comparisons, scorer version
   and human review provenance alongside the report.
4. Export with `tmf export --job <id> --report <technical.json>
   --quality-report <quality.json>`. Foundry checks the training-time contract,
   both report digests, job ID, suite/scorer hashes, checkpoint pointer hash,
   tuned weight fingerprint and every
   external gate. Missing, stale, edited or failing evidence blocks export;
   passing evidence is copied into the checksummed bundle.

The prior 20-step Gemma 4 smoke job failed its technical loss gate. It was
created before an external-quality contract and cannot be promoted using this
interface; use a fresh job ID after a new contract is frozen.
