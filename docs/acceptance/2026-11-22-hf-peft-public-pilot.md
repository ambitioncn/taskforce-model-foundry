# HF/PEFT repair and public-task pilot (2026-11-22)

This record covers the local TinyLlama 1.1B Chat (Apache-2.0) single-GPU LoRA
repair. It is **not** a production deployment or a full BANKING77 benchmark.
The source changes fix CUDA device selection, checkpoint-step resume, EOS
supervision, adapter/base lineage, one-load batch generation, device reporting,
and durable background evaluation attempts outside the Gateway cgroup when a
user systemd instance is available. Direct-launch fallback explicitly reports
`restartDurable=false`.

## Public customer-support intent pilot

- Data: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), CC BY 4.0. Official train/test CSV SHA-256:
  `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` /
  `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`.
- Before evaluating the model, choose the first eight alphabetically sorted
  categories with at least 100 train examples: `Refund_not_showing_up`,
  `activate_my_card`, `age_limit`, `apple_pay_or_google_pay`,
  `automatic_top_up`, `balance_not_updated_after_bank_transfer`,
  `balance_not_updated_after_cheque_or_cash_deposit`,
  `beneficiary_not_allowed`. Per class, select 50 official-train and 20
  official-test rows by ascending SHA-256 of query text. The official test
  split is not used for training or model selection.
- One candidate: 300 steps, LoRA rank 8/alpha 16 on `q_proj,v_proj`, learning
  rate 0.0002, batch size 1, CUDA float32, greedy generation. Foundry's
  training-source dedupe reduced 400 rows to 223, then split them 156/34/33
  for train/validation/holdout with zero measured cross-split leakage.
- Frozen pilot gates: official-test accuracy >=60%, absolute gain over the
  unmodified base >=20 percentage points, strict one-field JSON >=90%.
  Observed base **0/160**, tuned **144/160 (90%)**, JSON **160/160**.
  Foundry's separately gated internal holdout was **29/33**, 4/4 technical
  gates passed. Base model weight SHA-256 remained
  `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`;
  LoRA parameter L2 delta was 3.259838 and peak CUDA allocation 2434.503 MiB.
- General-regression screen: the first 40 ARC-Easy test items were
  inconclusive (both models emitted invalid single-letter answers). A separate
  80-item set, fixed by ranks 41–120 of SHA-256-sorted test IDs, used
  conditional answer-choice log likelihood. Base **15/80**, tuned **16/80**:
  no >10-point regression on this small screen, but low absolute scores do
  not certify broad ability. Five numeric-choice items required scoring their
  actual multi-token labels rather than assuming all choices were A–D.
- The gated local adapter export passed all 11 manifest checksums; bundle
  digest `e73a501f5dee33ccd81d1910a6e580adf9ff73d79043c612bbb9be6d2af2f967`.
  Fresh offline processes loaded original and exported adapters and matched
  generated token digests on 3/3 fixed prompts. The export still needs the
  separately supplied base model; no weights or datasets are included here.

## Isolated stability probes

Three sequential builtin-js jobs finished independently (`succeeded`, one
attempt each); replaying a finished ID did not rerun training. A separate HF
fixture job was killed with SIGKILL after a complete `checkpoint-50` appeared;
the original process exited -9 and a second attempt resumed from that exact
checkpoint to `succeeded`. Corrupt config/adapter files failed with nonzero
exit; a modified evaluation report was rejected by digest validation.
Targeted `ENOSPC` injection during an isolated job's data write produced
exit 1 and durable `failed` with error code `ENOSPC`, not a success receipt.
The OpenClaw background tool also demonstrated duplicate-request idempotency,
attempt-bound reports and explicit retry after interruption in the fixture
test and live plugin verification.

**Unverified boundaries:** actual host power loss, physical disk exhaustion,
running Gateway restart with a live worker, full 77-class BANKING77 quality,
and authorized de-identified enterprise data. SIGKILL and targeted `ENOSPC`
injection exercise failure paths, not those physical events. The public pilot
does not establish production or any other domain-specific model quality.

For a local fixture regression run with a supported Python environment:

```sh
TMF_PYTHON=/path/to/python TMF_DEVICE=cpu npm test
```

When the full HF stack is present, the test builds its small random-init
fixture offline if missing; it never downloads pretrained weights.
