# External quality gates for domain-specific models

Foundry's independent loss/strict-match report is a **technical** gate, not a
domain-quality judgement. A project may require separately scored quality
evidence before export. Foundry does not ship domain datasets, scorers, judges,
rubrics or promotion thresholds. This mechanism does not run a scorer or claim
that a checksum proves the scores are truthful.

1. **Before starting a new training job**, add `eval.externalQuality` to
   `tmf.config.json`. Freeze the complete suite and scorer files; put their
   SHA-256 digests in `suiteSha256` and `scorerSha256`. Declare nonempty gates
   using the same `id`, `metric`, `comparator`, `threshold`, optional
   `sampleMetric`/`minSamples` syntax as `eval.gates`. The training manifest
   and job input digest pin this contract. Do not retrofit it onto an old job.
   Optional `profile` names the project's protocol; optional `references` is a
   map of project-defined names to SHA-256 digests (for example a baseline
   output or rubric). Foundry stores them in the frozen contract without
   interpreting their domain meaning.
2. Evaluate the trained checkpoint with `tmf evaluate`. An external scorer can
   call `qualityContext({configFile, jobId, evalReport})` to get the frozen
   contract and checkpoint hash only after a passing technical report. Run the project's
   independently maintained suite and scorer only after its protocol and
   thresholds are frozen. Translate **all** required criteria into contract
   gates before looking at candidate results. Foundry supplies no default
   domain thresholds.
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

An old job created without the external-quality contract cannot be promoted
retroactively; use a new job ID after freezing the contract. Keep each
domain-specific recipe in its own project, outside the generic Foundry package.
