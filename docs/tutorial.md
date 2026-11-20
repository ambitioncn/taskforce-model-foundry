# First run: from install to a trained adapter

This tutorial takes about five minutes and needs nothing but Node.js 20+. It
uses a corpus the tool generates for you, so no private data, no API key and no
network access are involved. Every step below does real work: the training step
actually moves weights, and the job state it writes survives the process.

## 1. Install

From a clone:

```bash
git clone https://github.com/ambitioncn/taskforce-model-foundry.git
cd taskforce-model-foundry
npm install          # there are no runtime dependencies; this just links `tmf`
node bin/tmf.mjs version
```

Or as a dependency of your own project:

```bash
npm install taskforce-model-foundry
npx tmf version
```

The rest of this page writes `tmf`. From an install the binary lives in
`node_modules/.bin`, so write `npx tmf` (or add that directory to `PATH`); from
a source checkout without a link, write `node bin/tmf.mjs`.

## 2. Check the environment first

```bash
tmf doctor
```

`doctor` reports *every* check instead of stopping at the first failure, split
into `required` (a missing one makes the exit code 1) and `optional` (reported
as a warning — an unavailable GPU backend is normal on a laptop). `tmf doctor
--json` gives the same result as machine-readable JSON, which is what CI and
the OpenClaw/Hermes adapters consume.

```bash
tmf backends     # which trainer backends exist and which are usable here
```

## 3. Create a project

```bash
mkdir my-foundry && cd my-foundry
tmf init --count 160
```

This writes `tmf.config.json` plus a synthetic corpus under `data/`. `init`
refuses to overwrite an existing config (exit code 1), so re-running it cannot
silently destroy your settings. To regenerate only the corpus later:

```bash
tmf synth --out data --count 320
```

## 4. Validate the config before spending time on training

```bash
tmf config validate
```

Validation is exhaustive: you get one line per problem with the exact field
path (`train.mode`, `data.datasets[1].file`, …) rather than one error at a
time. The accepted `train.mode` values are constrained to what the selected
backend actually supports (`lora`, `pretrain`, `full-finetune`), so an
impossible run fails here instead of halfway through training.

## 5. Train

```bash
tmf train --job my-first-job
```

The job runs as four durable steps — `data -> base -> train -> finalize` — and
each one is checkpointed. Useful properties:

- **Resumable.** Re-run the same `--job` id and finished steps are reused
  instead of recomputed. `--no-resume` ignores previous state; a job an
  operator cancelled needs the explicit `--resume-cancelled`.
- **Auditable.** Every step and training step appends to an audit log.
- **Honest about results.** The final manifest reports the parameter L2 delta,
  so "it trained" is a measured claim, not a hope.

### Default hyperparameters

`tmf init` writes these, and they are the ones the tests pin:

| key | default | why |
| --- | --- | --- |
| `steps` | `120` | enough to see the loss move on a first run; raise it for real jobs |
| `lr` | `0.005` | `0.05` oscillates on the builtin-js AdamW path and the holdout improvement disappears |
| `batchSize` | `4` | keeps a CPU-only run inside the default wall-clock budget |

Raising `lr` back toward `0.05` is the single most common way to make a run
look trained while the evaluation gates reject it.

## 6. Inspect what happened

```bash
tmf status my-first-job
```

The state and audit trail are read back from disk, not from memory, so this
works from a new shell, after a reboot, or from another process. An unknown job
id exits 1.

## 7. Evaluate on held-out data

Evaluation is deliberately a separate job from training: it loads the base and
adapter checkpoints from disk, scores both on a holdout split that is asserted
to be disjoint from training data, and only then decides whether an improvement
may be claimed at all. `verdict.improvementClaimAllowed` is false unless the
gates pass, and `assertDeliverable(report)` throws `EVAL_NOT_DELIVERABLE` on a
failed gate, so a regression cannot be exported as a win.

The evaluation job is available programmatically as `evalJob.runEvaluationJob`
(see `test/eval-job.test.mjs` for a complete worked example, including the
refusal path).

## 8. Use it from code instead of the shell

The CLI is a thin wrapper over the programmatic API, so embedders get identical
behaviour without shelling out:

```js
import { doctor, synthesize, train, status } from 'taskforce-model-foundry';

const health = await doctor({ baseDir: process.cwd() });
if (!health.ok) throw new Error('environment is not ready');

await synthesize({ baseDir: process.cwd(), outDir: 'data', taskCount: 160 });
const { job, artifacts } = await train({ baseDir: process.cwd(), jobId: 'from-code' });
console.log(job.state, artifacts.metrics.parameterL2Delta);  // 'succeeded', e.g. 19.1

const st = await status({ baseDir: process.cwd(), jobId: 'from-code' });
console.log(st.found, st.state.state, st.audit.length);
```

Lower-level building blocks are re-exported under stable namespaces —
`data`, `synth`, `jobs`, `trainJob`, `evalJob`, `gates`, `backends` — so you can
compose your own job without importing internal file paths.

## 9. See the whole thing run

```bash
# from an install (the script ships in the tarball)
bash node_modules/taskforce-model-foundry/scripts/e2e-demo.sh
TMF_DEMO_KEEP=1 bash node_modules/taskforce-model-foundry/scripts/e2e-demo.sh

# from a source checkout
npm run demo          # bash scripts/e2e-demo.sh
TMF_DEMO_KEEP=1 npm run demo   # keep the temp workdir to poke at the artifacts
```

The demo runs in a throwaway directory and walks the full chain: version,
backends, init, config validate, doctor, train, status, and a resumed re-run.

## 10. Full-parameter runs instead of LoRA

`mode` accepts exactly three values (`src/config.mjs`, `MODES`):

| mode | what it trains | checkpoint written under `<runRoot>/<jobId>/model/` |
| --- | --- | --- |
| `lora` | low-rank adapter on top of a frozen base | `adapter.json` |
| `pretrain` | all parameters, from the seeded init | `tuned.json` |
| `full-finetune` | all parameters, starting from the base checkpoint | `tuned.json` |

Backends advertise which modes they accept, and the train job refuses the rest
*before* spending a step:

| backend | modes |
| --- | --- |
| `builtin-js` | `pretrain`, `lora`, `full-finetune` |
| `hf-peft` | `lora`, `full-finetune` |

```
$ tmf backends
```

Ask for a mode a backend does not support and the job fails fast with
`backend "<id>" does not support mode "<mode>"` (`src/jobs/train.mjs`), so a
typo costs you a second, not an afternoon.

To switch a project to a full fine-tune, change one field:

```json
{ "mode": "full-finetune" }
```

Then re-run `tmf config validate` and `tmf train`. Two consequences worth
knowing before you do:

- The deliverable is a whole model (`model/tuned.json`), not a small adapter,
  so exports are larger and no longer compose with a separately shipped base.
- `tmf evaluate` scores the tuned checkpoint instead of base+adapter; the
  baseline comparison in the report is still the untouched base checkpoint, so
  the gate still answers "did training help?".

## 11. Exporting against a report you already reviewed

By default `tmf export` re-evaluates, because an export is gated on evaluation.
When a human has already read a report and signed off on the numbers, pin the
bundle to *that* report instead of to a fresh run:

```
$ tmf export --job <jobId> --report .tmf-run/<jobId>/eval-report.json
```

Both arguments matter. Reports live under `<runRoot>/<jobId>/`, and `runRoot`
defaults to `.tmf-run` (`src/config.mjs`), not `runs/`. Pass `--job` as well:
without it the job id falls back to the project directory name, and export
fails with `no durable job state for "<dirname>"` before it ever reads your
report. Note also that `tmf evaluate` exits 1 when the gates fail, yet it
still writes `eval-report.json` — a report on disk is evidence of a scored
run, not proof that it passed.

The supplied report has to prove it is the same file that was produced
(`assertReportIntegrity`, `src/jobs/evaluate.mjs`). Export refuses it otherwise:

| situation | error |
| --- | --- |
| not a JSON object | `eval report must be an object` |
| wrong/absent `format` | `eval report format mismatch: expected <format>, got <got>` |
| no `reportDigest` | `eval report is missing reportDigest` |
| contents edited after the fact | `eval report digest mismatch: file claims <a>, contents hash to <b>` |
| report belongs to another run | `eval report belongs to job <other>, not <jobId>` |

That last check is the one that matters in practice: it stops a bundle from
claiming the metrics of a different, better job. Editing a single number in the
report file — even a comment-like string — breaks the digest, so a pinned
export is only as good as the report nobody touched.

Programmatically the same path is `exportModel({ evalReport })`; passing
`evaluateIfMissing: false` with no report is an error rather than a silent
re-evaluation (`src/api.mjs`).

## 12. Rolling back

Nothing here mutates a previous run. Every job owns an immutable directory
`<runRoot>/<jobId>/` holding its `model/`, its `eval-report.json`, its job
state and its audit events, and `tmf export` writes into
`<runRoot>/<jobId>/export/` unless you pass `--out`. Rolling back is therefore
re-pointing at an older job, not undoing a newer one.

1. Find the job you want to go back to and confirm it is actually finished:

   ```
   $ tmf status <oldJobId>
   ```

2. Re-export it from its own reviewed report, so the bundle you ship is byte-
   for-byte justified by the evaluation you trust:

   ```
   $ tmf export --job <oldJobId> --report .tmf-run/<oldJobId>/eval-report.json --out dist/rollback
   ```

   The digest and `lineage.jobId` checks from section 11 are what make this
   safe: if you accidentally hand it the newer run's report, export fails
   instead of producing a mislabelled bundle.

3. Point the consumer at the rolled-back bundle. For the OpenClaw / Hermes
   integrations that means the bundle path in the host config — see
   [docs/integration.md](./integration.md); no state inside the toolkit needs
   to be rewound.

To abandon a bad run rather than roll back to a good one, leave its directory
in place and stop referencing it. Deleting `<runRoot>/<jobId>/` is allowed but
destroys the audit trail for that job, and any bundle already exported from it
keeps its own copy of the report, so deletion is not a recall.

## Troubleshooting

| Symptom | What it means |
| --- | --- |
| `tmf doctor` exits 1 with `node-version` | Node is older than 20; upgrade. |
| `tmf doctor` exits 1 with `trainer-backend` | No backend is usable here; `tmf backends` shows why. |
| `tmf init` exits 1 | A config already exists; edit it or pick another directory. |
| `tmf config validate` lists dataset problems | A `data.datasets[].file` path is missing — `tmf synth` regenerates the corpus. |
| `tmf status <id>` exits 1 | No job with that id in this project directory. |
| Exit code 2 from any command | Usage error (unknown command or flag); `tmf help` lists the surface. |

## What is *not* in scope here

Private company data, paid training runs, publishing and production changes are
each separately authorized operations. This tutorial stays entirely on
self-generated synthetic data and local CPU training.
