# Bounded autonomous campaigns

Foundry can advance a **preapproved** sequence of model candidates without an
agent harness or an active chat session. The campaign controller uses the core
API, durable job receipts and an independent background worker. It chooses
whether to continue or stop from frozen technical gates; it does not generate
new datasets, invent model configurations, lower thresholds, export, publish or
claim business acceptance.

## Manifest

Write a `campaign.json` beside all candidate configs and their datasets:

```json
{
  "format": "tmf-campaign-v1",
  "id": "ticket-pilot-r1",
  "candidates": [
    { "id": "small-lr", "configFile": "candidate-small-lr.json" },
    { "id": "larger-lr", "configFile": "candidate-larger-lr.json" }
  ],
  "budget": {
    "maxTrials": 2,
    "maxTotalTrainSteps": 400,
    "deadlineAt": "2026-12-01T00:00:00Z"
  },
  "policy": {
    "stopOnTechnicalPass": true,
    "rankMetric": "tuned.holdout.loss",
    "direction": "min"
  }
}
```

Each candidate is a normal Foundry config and must explicitly declare
`eval.gates`; demo defaults are rejected. Candidates must have the same
project, backend, mode, model, dataset definitions and bytes, and technical
and external-quality contracts. Hyperparameters and seed may differ. Before
execution the controller fingerprints the manifest, configs and datasets; any
post-hoc edit makes replay fail instead of silently changing the experiment.
`maxTotalTrainSteps` charges each attempted candidate's declared `hyper.steps`
conservatively. The UTC deadline prevents starting another candidate after it;
active training/evaluation is not forcibly killed by that deadline, so each
candidate must also set its own resource/time budgets.

## Run without a harness

```bash
tmf campaign start --plan campaign.json --json
tmf campaign status --plan campaign.json --json
tmf campaign cancel --plan campaign.json
tmf campaign start --plan campaign.json --resume-cancelled
```

`start` is idempotent. It prefers a transient user systemd unit on Linux,
falling back to a detached process where systemd is unavailable. The worker
persists a campaign state, audit log, per-candidate job states and evaluation
reports. If the worker exits unexpectedly, `status` reports `interrupted`;
run `start` again to continue from completed step receipts. For a supervised
foreground run use `tmf campaign run --plan campaign.json`.
The direct fallback is not guaranteed to survive a host's cgroup shutdown;
use a service manager for that durability requirement. A failed campaign is
terminal and requires a new campaign id after the underlying issue is fixed.

A campaign decision of `technical_passed` only means its predeclared technical
gates passed. If the configs require external-quality evidence, the decision is
`awaiting_external_quality`. Neither is a business acceptance verdict.
No candidate is automatically exported. Review the reports, supply any
predeclared external-quality evidence, then use the normal gated export path.

Harnesses may call `tmf_campaign_start`, `tmf_campaign_status` and
`tmf_campaign_cancel` through the optional adapter. The campaign continues in
the Foundry worker; an agent harness is not its scheduler or source of truth.
