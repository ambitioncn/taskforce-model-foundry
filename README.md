# taskforce-model-foundry

An automated model customization engine for OpenClaw and Hermes: it generates or
ingests a corpus, runs **real** post-training (weights actually move), evaluates
the result on an independent holdout split, and refuses to call a regression an
improvement.

- Node.js 20+, **zero runtime dependencies**
- Real training, not a simulation: every job reports its parameter L2 delta
- Durable, resumable, audited jobs — state is re-read from disk, not memory
- Gate-guarded evaluation: `assertDeliverable()` throws on a failed gate
- No private data and no network needed to try it out

## Quickstart

```bash
git clone https://github.com/ambitioncn/taskforce-model-foundry.git
cd taskforce-model-foundry
node bin/tmf.mjs --version
node bin/tmf.mjs init -C /tmp/my-foundry
node bin/tmf.mjs train -C /tmp/my-foundry --job first
node bin/tmf.mjs status first -C /tmp/my-foundry
```

This release is distributed as source on GitHub; an npm registry release has not
been published. The built-in CPU backend needs no npm dependencies.

See the full walkthrough in **[docs/tutorial.md](docs/tutorial.md)**, the host-side
contract in **[docs/integration.md](docs/integration.md)**, or watch the
whole chain run end to end:

```bash
# from an install (the script ships in the tarball)
bash node_modules/taskforce-model-foundry/scripts/e2e-demo.sh

# from a source checkout
npm run demo              # bash scripts/e2e-demo.sh
```

The demo writes to a temporary workdir and removes it again, so it is safe to
run against an installed copy.

## Programmatic API

```js
import { doctor, synthesize, train, status } from 'taskforce-model-foundry';
```

The CLI is a thin wrapper over this API, so embedders get identical behaviour
without shelling out. Lower-level building blocks are exported under the stable
namespaces `data`, `synth`, `jobs`, `trainJob`, `evalJob`, `gates`, `backends`.

## CLI surface

| Command | Purpose |
| --- | --- |
| `tmf doctor` | Check environment, backends and config (`--json` for CI) |
| `tmf init` | Write a starter `tmf.config.json` plus synthetic data |
| `tmf config validate` | Report *every* config problem with its field path |
| `tmf backends` | List trainer backends and their availability |
| `tmf synth` | Generate a synthetic corpus |
| `tmf train` | Run a real post-training job (resumable, cancellable with `Ctrl-C`) |
| `tmf status <jobId>` | Durable job state and recent audit events |

Exit codes: `0` ok, `1` check/job failed, `2` usage error, `130` job cancelled
by an operator signal (durable state kept; the printed `--resume-cancelled`
command continues it).

Two trainer backends ship: `builtin-js` (pure JS, CPU, no extra deps) and
`hf-peft` (local transformers + PEFT subprocess, no downloads, no hosted API).
`hf-peft` train / checkpoint-reload / independent baseline-vs-tuned evaluation
/ generation are exercised for real on CPU against a random-init local fixture
(`scripts/make-tiny-hf-fixture.py` + `scripts/hf-peft-e2e.mjs`); CUDA and
reduced precision are **not** verified here, and its per-worker RLIMIT bounds
are resource caps, not a sandbox. The capability table in
[docs/tutorial.md](docs/tutorial.md) marks each item measured, unverified or
not supported. `hf-peft` implements the same backend contract as `builtin-js`
(including `fingerprint()` and in-memory `examples`).

Tokenizer ownership is a backend capability, not a job assumption:
`capabilities.ownsTokenizer` plus `prepareDataset()` is the single place that
decides whether ids come from a vocabulary the job built from the data split
(`builtin-js`) or from the model's own tokenizer inside the Python worker
(`hf-peft`). Both the training job and the evaluation split loader go through
it, so `tmf train` is no longer builtin-only. Measured on CPU against the
random-init local fixture (`docs/reports/g12d-cli-e2e-evidence.json`):
`tmf train` succeeds (16 steps, loss 3.4277 -> 3.3744, parameter L2 delta
0.652, step checkpoints written), `tmf status` reports the durable state, and
`tmf evaluate` scores baseline and tuned in two independent processes (tuned
holdout loss 3.9% better).

Cancel and resume are stated at the precision the evidence supports.
`Ctrl-C` or `SIGTERM` during `tmf train` aborts the job's `AbortSignal`: the
job stops between steps, persists `cancelled` with the steps it did finish,
prints the exact resume command and exits `130`; a second signal exits at once
and says the in-flight step is lost. `tmf train --resume-cancelled` then re-runs
only the steps without a durable receipt -- measured on job `g12d-cancel4`,
which cancelled after `data`,`base` and resumed `train`,`finalize` to
`succeeded` in 15s against ~40s cold
(`docs/reports/g12d-cancel-resume-evidence.json`). This is **job-step** resume,
not training-step resume: a cancelled `train` step restarts from the last
trainer checkpoint. Cancellation is **cooperative** -- the Python worker is
`SIGKILL`ed, but bounded-latency forced abort of an arbitrary backend step is
not offered, and there is no rollback beyond the step receipts.

Gate thresholds are project policy, declared under `eval.gates`, and the
built-in list is only the default. The defaults were calibrated against the
`builtin-js` demo, so a random-init tiny fixture cannot clear them -- the first
`hf-peft` export run stopped at `EXPORT_BLOCKED` with exact match 0 after 16
steps, which is the gate working rather than the export path failing. Declaring
`eval.gates` replaces the default list wholesale instead of merging it, because
a half-overridden gate set silently drops checks; an empty list, a duplicate
gate id, an unknown comparator and a `minSamples` without its `sampleMetric`
are all config errors reported by path. A project that narrows its gates is
stating, on the record, which quality bar its deliverable was actually held to.

`tmf doctor` grades an absent or unreadable config as a *warning* and still
exits `0`, so the environment and backend verdict survives a project that has
no config yet. Point it at the file with `--config <file>`, or use `tmf config
validate`, when CI must fail on a bad config.

## Scope and authority

Synthetic data and local CPU training are the default. Real private company
data, paid training runs, production changes and publishing are each separately
authorized operations and are never performed implicitly.

## Development

```bash
npm test                  # the full suite
```

MIT licensed.
