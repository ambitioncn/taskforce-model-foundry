# taskforce-model-foundry — gap-closure final report

Scope: close the real gaps between the original development plan and the shipped
open-source code, with independent evidence per item. Isolated worktree, isolated
Python venv, synthetic data only, random-init local tiny HF fixture, bounded CPU.
Validation used no paid compute, private data or credentials. This report describes
the pre-publication validation; source publication is a separate authorized step.
Raw machine-specific receipts are retained locally, not included in the repository.

Per-item: plan item → change → actual measured result → remaining limit.

## Theme 1 — hf-peft was not actually implemented

| id | plan item | change | actual result | remaining limit |
| --- | --- | --- | --- | --- |
| G1 | `evaluate()`/`generate()` must work when deps are present | removed the unconditional throw; both run in the Python worker | real CPU runs return holdout loss and decoded continuations (`hf-peft-e2e-evidence.json`) | CPU float32 only |
| G2 | complete dependency probe | `probe()` now checks torch, transformers, peft, **datasets**, accelerate | the missing-package reason string lists all five (see installed-CLI `doctor` in `g12d-pack-install-evidence.json`) | probe checks importability, not version compatibility |
| G3 | real training, not a rendered script | worker returns loss history + `parameterL2Delta` | loss moves and weights move on every recorded run (e.g. 3.4277→3.3744, L2 delta 0.652) | tiny random-init fixture; absolute loss values are meaningless |
| G4 | independent baseline vs tuned eval | two separate `evaluate()` processes | tuned holdout loss 3.9% better than baseline, scored in distinct processes, not two reads of one training log | improvement magnitude is fixture-specific |
| G5 | checkpoint save/reload | `saveModel` → fresh-process `loadModel` → `evaluate` | `adapterLoaded: true` on reload in a new process | — |
| G6 | `full-finetune` mode | exercised `train_full_finetune` | real all-weights run with non-zero weight movement | — |
| G10 | offline/local-only | verified no egress needed | runs against a locally generated fixture (`scripts/make-tiny-hf-fixture.py`) | — |
| G11 | the Python worker must ship | packaging re-check | `package/src/train/backends/hf_peft_runner.py` is in the tarball (50 files) | — |
| G13 | evidence receipts must not ship inside the package | `files`/ignore corrected | tarball no longer carries `docs/reports/` receipts | — |

## Theme 2 — resource bounds, cancel and resume vs the original 8 milestones

| id | plan item | change | actual result | remaining limit |
| --- | --- | --- | --- | --- |
| G7 | `supportsResume: true` must be backed | resume from the last `checkpoint-<step>` dir | real resume verified; steps after the checkpoint are **replayed**, not recovered | step-coarse, not exact-state resume |
| G8 | `supportsCancel: true` must be backed | cancellation path added; worker `SIGKILL` | cancel verified; in-flight work lost, written checkpoints survive | **cooperative** — no bounded-latency forced abort |
| G9 | budgets covered only steps/tokens/time/cost — no memory bound or isolation | added `maxMemoryMb` (`RLIMIT_AS`) and `maxCpuSeconds` (`RLIMIT_CPU`) + thread pinning on the worker | negative test: exceeding the bound terminates the worker with `BACKEND_MEMORY_LIMIT`; a bound below the framework footprint fails at import, not mid-step | these are **caps, not a sandbox**: no filesystem or network isolation. Stated as `isolation: "process-rlimit"` |

## Theme 3 — contract parity and the `tmf train` job path (G12)

| id | plan item | change | actual result | remaining limit |
| --- | --- | --- | --- | --- |
| G12a | same backend contract as `builtin-js` | added `fingerprint()` | path-independent sha256; moving a dir keeps the digest, one changed weight byte changes it | — |
| G12b | same `train()` parameter shape | in-memory `examples` materialized to `train-examples.jsonl` | 8 real CPU steps with non-zero `parameterL2Delta` (`g12-contract-verify.mjs`) | — |
| G12c | export must be self-contained | `exportModel()` embeds the referenced weights dir and hashes every file into the manifest | bundle carries the adapter bytes (extension-filtered, depth-bounded, 512 MB cap) | the **base** model is referenced by `modelId`, not copied — a consumer needs the same local base |
| G12d | `tmf train` must not be builtin-only | tokenizer ownership moved to the backend (`capabilities.ownsTokenizer` + `prepareDataset()`); fixed `createModel` shape, `attachAdapter`, stale post-train handle, and the label-span merge bug | full CLI loop on hf-peft: `train` → `status` → `evaluate` → `export` (`g12d-cli-e2e-evidence.json`, `g12d-export-evidence.json`) | — |
| G12d-gates | configurable eval gates | removed dead `resolved.eval?.gates` fallback; `eval.gates` validated by path | declaring gates replaces the default list wholesale; the first hf-peft export honestly stopped at `EXPORT_BLOCKED` | default gate thresholds are calibrated for the builtin demo, not for a tiny fixture |
| G12d-cancel | cancel/resume must be reachable for a CLI operator | `tmf train` installs SIGINT/SIGTERM handlers feeding the job `AbortSignal`; catches `JobCancelledError`, prints durable state + the exact resume command, exits **130** | job `g12d-cancel4`: cancelled 07:18:35.758Z after `data`,`base` (audit `job.cancelled`), then `--resume-cancelled` reached `succeeded` in 15s vs ~40s cold; `g12d-cancel5` re-verified exit 130 | **job-step** resume, not training-step resume. Measured basis: 240 LoRA steps = 0.883s wall clock while Python startup/model load dominate, so a shell signal cannot land inside the step loop |

## Final verification against the submitted tree

- `npm test`: 234 tests, 231 pass, 0 fail, 3 skipped, exit 0 — rerun **after** the
  `bin/tmf.mjs` cancel change and the README/tutorial edits.
  The 3 skips are the hf-peft integration tests on the *system* python, which has
  no ML stack; the same behaviour runs for real in the isolated venv.
- `npm pack` + install into a clean temp project: tarball installs, `tmf doctor --json`
  exits 0, `tmf help` documents exit 130, and `tmf init && tmf train` ran a real
  builtin-js job to `succeeded` from the installed package.
  Receipt: `docs/reports/g12d-pack-install-evidence.json`.
- Docs now separate **measured / unverified / not supported** per capability
  (`docs/tutorial.md` capability table) and state the cancel/resume semantics at the
  precision the evidence supports (`README.md`).

## What is explicitly NOT claimed

CUDA, bf16/fp16, multi-GPU/distributed/hosted training, filesystem or network
isolation, forced bounded-latency abort, training-step-granular resume, and any
production deployment. Nothing here was committed, pushed or published.

## Follow-up validation

The system Python 3.14 ML dependencies were subsequently installed. Running
`TMF_PYTHON=/usr/bin/python3 node --test test/train-hf-peft.test.mjs` passed
7/7 tests with zero skips, including the three previously skipped integrations.

Publication validation on the final source tree: system Python 3.14,
`TMF_PYTHON=/usr/bin/python3 npm test`: **234 passed, 0 failed, 0 skipped**.
`npm pack --dry-run` and `git diff --cached --check` also passed.
