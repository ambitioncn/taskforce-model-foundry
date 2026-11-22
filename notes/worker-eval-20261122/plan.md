# Foundry long-job and screenplay-eval repair

Goal: make 31B local jobs durable and efficient, report truthful budgets/tokens, and allow a separately frozen screenplay quality verdict without weakening Foundry gates.

Constraints: work only in isolated branch; keep installed 0.2.0 unchanged until tests and runtime proof; no RSI restart, private-data egress, or model publication. Existing 20-step smoke failed a predeclared loss gate and must remain failed.

Phases:
1. Reuse async-evaluate request/attempt lifecycle for CLI background train/eval; status and explicit retry, no duplicate execution.
2. For HF, evaluate holdout loss, generation and validation loss in one model load per side; preserve exact metric semantics and ordering.
3. Align model-specific virtual memory/time budgets across operations; replace fabricated tokensSeen with measured or unknown, recording data token count separately.
4. Add external quality-report binding/gate interface for frozen Chinese screenplay 120-question evaluation; independent report must be hash/job bound.
5. Targeted tests, full suite, tiny live proof; only then consider 31B live proof, never re-train or relax the failed gate by default.

Verification: unit tests for request dedupe/retry, evaluation metric parity, config validation and report lineage; CLI help/status smoke; full npm test; isolated small HF live run. Report counts and limits.
