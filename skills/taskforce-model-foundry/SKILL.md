---
name: taskforce-model-foundry
description: Train, evaluate, resume, and export local models with Taskforce Model Foundry. Use for tmf CLI jobs, HF/PEFT fine-tuning, independent quality gates, and OpenClaw or Hermes model-customization workflows.
license: MIT
---

# Taskforce Model Foundry

1. Establish the job contract: dataset provenance, allowed compute host, backend,
   model path, resource/time budget, independent holdout, and quality thresholds.
   Default to synthetic data and local CPU. Obtain task-specific authorization
   before private data, paid compute, remote transfers, production changes, or
   publishing. Finish with a recorded scope and success criteria.

2. Locate a trusted installation using [the command reference](references/commands.md).
   Run version, doctor, and backends checks. Doctor can return success while a
   missing config is only a warning: validate the actual config separately.
   Finish with the executable, version, usable backend, and validated config known.

3. Prepare an isolated project directory. Use `init` only for a new synthetic
   project; for an existing project inspect its configuration before changing it.
   Keep data outside this skill directory. For HF/PEFT select an existing local
   model and a dedicated Python environment; inspect `docs/tutorial.md` in the
   installation for current config fields. Do not invent model download or GPU
   setup commands. Finish with licensed/provenanced data and disjoint splits.

4. Fix the evaluation policy before training. `eval.gates` replaces the default
   list in full; validate it, and preserve the approved thresholds. A tiny fixture
   with fixture-specific gates validates plumbing, not business quality. Finish
   with a recorded baseline, holdout policy, and explicit gate list or defaults.

5. Assign a stable job ID, inspect durable status, then train. Prefer
   `--background` for long local jobs and retain the returned request ID; use
   `train-status`/`evaluate-status` for receipts and `train-cancel` to cancel a
   background train. After a timeout or disconnected session, query durable
   state before any explicit `--retry`; never duplicate a running request.
   Foreground cancellation uses SIGINT/SIGTERM and exit 130. Resume an
   intentionally cancelled job only when authorized to continue. Finish with
   durable succeeded, cancelled, or failed state and the actual process exit
   code, not just a launched-process receipt.

6. Evaluate baseline and tuned model on the independent holdout. Check leakage,
   sample counts, actual parameter changes, and each declared gate. Treat failed
   evaluation as a failed deliverable; diagnose it instead of weakening gates
   after seeing results. Finish with the report path, metrics, and gate verdict.

7. Export only a passing job, preferably pinning its reviewed evaluation report.
   Verify `checksums.txt` from the bundle directory. HF/PEFT exports include adapter
   bytes but reference the local base model: record its identity and availability.
   Finish with bundle path, digest, lineage, checksum results, and base dependency.

   A project may freeze an `eval.externalQuality` contract before training and
   supply its own scorer evidence with `--quality-report`; Foundry verifies the
   declared metrics, lineage and thresholds but does not ship domain-specific
   suites, rubrics or judges. Never infer business quality from a technical
   loss gate alone.

8. Report measured results separately from unsupported or unverified capabilities.
   Read [capability and host boundaries](references/boundaries.md) before making
   GPU, remote execution, recovery, or production claims. Include test pass/fail/
   skip counts; a skipped test is not a pass. Publishing requires its own actual
   registry/release and installation verification. Finish with evidence paths,
   remaining limits, and whether the artifact is local, published, or deployed.

9. For a multi-phase Foundry objective, track the **overall acceptance contract**
   separately from individual job states. A training checkpoint, technical-eval
   pass, diagnostic, or one candidate's rejection is phase progress, not overall
   completion. Before marking a campaign or continuation `done`, verify every
   frozen required gate and independent quality review with durable evidence.
   If authorized work remains, record progress and schedule the next bounded
   phase on the same continuation; do not let a one-shot check expire after an
   intermediate result. If deadline, budget, or missing external evidence prevents
   further work, report blocked/needs-attention with the precise blocker and
   resume condition. Never represent a partial technical pass as model acceptance.
