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

## Runnable Chinese screenplay v1 profile

The `screenplay-zh-v1` profile reuses the project's rules-v2 scorer, G5
promotion table, and order-swapped blind A/B tally scripts. They ship under
`scripts/screenplay-quality/`; their combined digest is frozen with the suite.
It implements **all** original promotion criteria: constrained-rule gain vs
DAPT at least 5 percentage points, scene completion at least 95%, broad/long
early-stop proxy at most 5% and no worse than base, no major category drop
greater than 5 points, two independent judges each rating 120 questions in
both A/B orders with win rate at least 55% and bootstrap lower bound above
50% in each of the four cells, and clean-validation loss regression vs DAPT
at most 2%. The 120 outputs and raw judge verdicts remain reviewable.

1. Before a **new** job, freeze the existing 120-question suite, full base and
   DAPT generation files, and DAPT clean-validation loss report:

   ```sh
   tmf quality-contract --suite /path/screenplay_zh_eval_v1.json \
     --base-outputs /path/base.jsonl --dapt-outputs /path/dapt.jsonl \
     --clean-dapt-report /path/dapt-clean-val.json --out contract.json
   ```

   Copy the resulting object into `eval.externalQuality` in `tmf.config.json`
   **before training**. The job manifest freezes it. The reference outputs must
   have all 120 IDs, exact suite prompts and recorded token-budget telemetry.
2. Train and pass the independent technical evaluation first. Generate the
   candidate's 120 responses with the same frozen protocol and record each
   `id`, `prompt`, `output`, `new_tokens`, `max_new_tokens`, and
   `hit_max_new_tokens`. Then stage the quality evidence:

   ```sh
   tmf quality-prepare --config tmf.config.json --job NEW_JOB \
     --report /path/technical-eval.json --suite /path/screenplay_zh_eval_v1.json \
     --base-outputs /path/base.jsonl --dapt-outputs /path/dapt.jsonl \
     --candidate-outputs /path/candidate.jsonl \
     --clean-dapt-report /path/dapt-clean-val.json --clean-dapt-model DAPT_NAME \
     --out /path/quality-stage
   ```

   This writes rules-v2 scores, the promotion table, and anonymized order-swapped
   pairs. Give judges only the blind pairs, **not** `blind/key.json`.
3. Collect raw verdict JSONL from two independent judges. Each judge file
   must have 240 rows (`id`, `order` = `order1`/`order2`, `verdict` = `A`/`B`/`tie`).
   Run the same clean-validation loss protocol on the candidate, then finalize:

   ```sh
   tmf quality-finalize --config tmf.config.json \
     --stage /path/quality-stage/stage.json \
     --judge judge1=/path/judge1.jsonl --judge judge2=/path/judge2.jsonl \
     --clean-candidate-report /path/candidate-clean-val.json \
     --clean-candidate-model CANDIDATE_NAME --out /path/quality-report.json
   tmf export --config tmf.config.json --job NEW_JOB \
     --report /path/technical-eval.json \
     --quality-report /path/quality-report.json --out /path/export
   ```

   Missing verdicts, altered source files, weaker thresholds, failed technical
   gates, or failed quality gates block promotion. These digests provide
   integrity and lineage checks, **not** authenticity of external judge or
   generation claims; retain generation logs and judge provenance for review.

The earlier 20-step Gemma 4 smoke job failed its technical loss gate and was
created before this contract. It cannot be retroactively promoted. A new
technically passing job, full DAPT reference outputs, and genuine two-judge
verdicts are required before reporting business quality acceptance.
