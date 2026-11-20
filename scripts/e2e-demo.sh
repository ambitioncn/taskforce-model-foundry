#!/usr/bin/env bash
# Repeatable end-to-end demonstration of taskforce-model-foundry.
#
# Everything here is real work: a synthetic corpus is generated, a model is
# actually post-trained (weights move), and the job state it leaves behind is
# re-read from disk afterwards. No private data and no network are involved.
#
# Usage:
#   bash scripts/e2e-demo.sh [workdir]
#   TMF_DEMO_KEEP=1 bash scripts/e2e-demo.sh    # keep the workdir for inspection
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMF="node $REPO/bin/tmf.mjs"
WORKDIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/tmf-demo-XXXXXX")}"
mkdir -p "$WORKDIR"

finish() {
  status=$?
  if [ "${TMF_DEMO_KEEP:-0}" = "1" ] || [ "$status" -ne 0 ]; then
    echo "workdir kept for inspection: $WORKDIR"
  else
    rm -rf "$WORKDIR"
    echo "workdir removed (set TMF_DEMO_KEEP=1 to keep it)"
  fi
}
trap finish EXIT

step() { echo; echo "=== $* ==="; }

step "0. version and backends"
$TMF version
$TMF backends

step "1. init: starter config + synthetic corpus (no private data)"
$TMF init -C "$WORKDIR" --count 160

step "2. config validate: every problem at once, with field paths"
$TMF config validate -C "$WORKDIR"

step "3. doctor: environment, backend and dataset self-check"
$TMF doctor -C "$WORKDIR"

step "4. train: a real post-training job (deterministic seed from the config)"
$TMF train -C "$WORKDIR" --job demo-job

step "5. status: durable job state and audit trail re-read from disk"
$TMF status -C "$WORKDIR" demo-job

step "6. re-run train: resumes the finished job instead of redoing the work"
$TMF train -C "$WORKDIR" --job demo-job

step "demo complete"
echo "Trained artifacts and job state live under: $WORKDIR"
echo "Independent held-out evaluation and its gate verdict are exercised by the"
echo "test suite (test/eval-job.test.mjs, test/eval.test.mjs): npm test"
