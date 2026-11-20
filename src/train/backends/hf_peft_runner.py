#!/usr/bin/env python3
"""Real local execution runner for the hf-peft backend.

This file is the backend's only execution path. It is a checked-in program
rather than a string rendered at call time so that what runs is what can be
reviewed. It is driven by a JSON spec file and prints exactly one result line
prefixed with ``__TMF_RESULT__``; everything else on stdout/stderr is framework
noise that the Node side keeps as a log artifact.

Operations: ``train``, ``evaluate``, ``generate``.

Resource bounds are applied in-process before any framework import does real
work: address-space (RLIMIT_AS) and CPU-seconds (RLIMIT_CPU) are hard kernel
limits, so exceeding them terminates this process rather than relying on the
caller noticing. Thread count is pinned so CPU use stays bounded. The step and
wall-clock budgets are enforced by a Trainer callback, which is cooperative by
construction and is reported as such.
"""

import json
import math
import os
import resource
import sys
import time

RESULT_PREFIX = "__TMF_RESULT__"


def emit(payload):
    sys.stdout.write(RESULT_PREFIX + " " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def apply_limits(spec):
    applied = {}
    mem_mb = spec.get("maxMemoryMb")
    if mem_mb:
        nbytes = int(mem_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        applied["maxMemoryMb"] = int(mem_mb)
    cpu_s = spec.get("maxCpuSeconds")
    if cpu_s:
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s)))
        applied["maxCpuSeconds"] = int(cpu_s)
    threads = int(spec.get("threads", 1) or 1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        os.environ.setdefault(var, "1" if var == "TOKENIZERS_PARALLELISM" else str(threads))
    applied["threads"] = threads
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    applied["offline"] = True
    return applied


def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_row(tok, row, max_len):
    prompt_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
    full_ids = tok(row["prompt"] + row.get("completion", ""), add_special_tokens=False)["input_ids"]
    full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[min(len(prompt_ids), len(full_ids)):]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def build_model(spec, torch, want_training):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = spec["modelId"]
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    adapter = spec.get("adapterDir")
    mode = spec.get("mode", "lora")
    peft_info = {"mode": mode, "adapterLoaded": False}
    if adapter and os.path.isdir(adapter):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=want_training)
        peft_info["adapterLoaded"] = True
    elif want_training and mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora = spec.get("lora") or {}
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora.get("rank", 8)),
                lora_alpha=int(lora.get("alpha", 16)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=lora.get("targets") or ["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    model.config.use_cache = False if want_training else True
    return tok, model, peft_info


def trainable_snapshot(model, torch):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def l2_delta(before, after, torch):
    total = 0.0
    for name, tensor in before.items():
        if name in after:
            total += float(torch.sum((after[name] - tensor) ** 2))
    return math.sqrt(total)


def op_train(spec):
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    torch.manual_seed(int(spec.get("seed", 1)))
    started = time.time()
    tok, model, peft_info = build_model(spec, torch, want_training=True)
    rows = load_rows(spec["examplesFile"])
    if not rows:
        raise ValueError("training needs at least one example")
    max_len = int(spec.get("maxSeqLen", 128))
    ds = Dataset.from_list([encode_row(tok, r, max_len) for r in rows])

    budget = spec.get("budget") or {}
    max_seconds = budget.get("maxSeconds")
    stop_state = {"reason": None}

    class BudgetCallback(TrainerCallback):
        """Cooperative stop: checked between steps, never mid-kernel."""

        def on_step_end(self, args, state, control, **kwargs):
            if max_seconds and (time.time() - started) > float(max_seconds):
                stop_state["reason"] = "time_budget"
                control.should_training_stop = True
            return control

    out_dir = spec["outDir"]
    os.makedirs(out_dir, exist_ok=True)
    save_steps = int(budget.get("checkpointEvery", 0) or 0)
    args_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(spec.get("batchSize", 2)),
        learning_rate=float(spec.get("lr", 1e-3)),
        max_steps=int(budget.get("maxSteps", 20)),
        logging_steps=1,
        report_to=[],
        save_strategy="steps" if save_steps > 0 else "no",
        seed=int(spec.get("seed", 1)),
        use_cpu=True,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    if save_steps > 0:
        args_kwargs["save_steps"] = save_steps
    # TrainingArguments gains and drops keyword arguments between transformers
    # major versions. Filtering against the real signature keeps this runner
    # working on the installed version instead of pinning one, and the dropped
    # names are reported so a silently ignored budget is visible.
    import inspect

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in args_kwargs if k not in supported)
    args = TrainingArguments(**{k: v for k, v in args_kwargs.items() if k in supported})
    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator, callbacks=[BudgetCallback()])

    before = trainable_snapshot(model, torch)
    resume = spec.get("resumeFrom")
    result = trainer.train(resume_from_checkpoint=resume if resume else None)
    after = trainable_snapshot(model, torch)

    history = [
        {"step": int(e["step"]), "loss": float(e["loss"])}
        for e in trainer.state.log_history
        if "loss" in e and "step" in e
    ]
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    checkpoints = sorted(
        os.path.join(out_dir, d) for d in os.listdir(out_dir) if d.startswith("checkpoint-")
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        "backend": "hf-peft",
        "mode": peft_info["mode"],
        "resumedFrom": resume or None,
        "steps": int(trainer.state.global_step) - (int(trainer.state.global_step) - len(history) if resume else 0),
        "totalSteps": int(trainer.state.global_step),
        "firstLoss": history[0]["loss"] if history else None,
        "finalLoss": history[-1]["loss"] if history else None,
        "bestLoss": min((h["loss"] for h in history), default=None),
        "trainLossMean": float(result.training_loss) if result and result.training_loss is not None else None,
        "trainableParams": trainable,
        "parameterL2Delta": round(l2_delta(before, after, torch), 8),
        "tokensSeen": int(sum(len(r["input_ids"]) for r in ds)) * max(1, int(trainer.state.global_step) // max(1, len(ds))),
        "wallClockSeconds": round(time.time() - started, 3),
        "peakRssMb": peak_rss_mb(),
        "stepBudget": int(budget.get("maxSteps", 20)),
        "stopReason": stop_state["reason"] or "completed",
        "unsupportedTrainerArgs": dropped,
    }
    emit({
        "op": "train",
        "metrics": metrics,
        "history": history,
        "adapterDir": adapter_dir,
        "checkpoints": checkpoints,
        "limits": spec.get("_appliedLimits"),
    })


def op_evaluate(spec):
    import torch

    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    tok, model, peft_info = build_model(spec, torch, want_training=False)
    model.eval()
    rows = load_rows(spec["examplesFile"])
    if not rows:
        raise ValueError("evaluation needs at least one example")
    max_len = int(spec.get("maxSeqLen", 128))
    total_loss = 0.0
    total_tokens = 0
    started = time.time()
    with torch.no_grad():
        for row in rows:
            enc = encode_row(tok, row, max_len)
            if len(enc["input_ids"]) < 2:
                continue
            ids = torch.tensor([enc["input_ids"]])
            labels = torch.tensor([enc["labels"]])
            out = model(input_ids=ids, labels=labels)
            # Count only supervised positions so the mean is token-weighted.
            counted = int((labels[:, 1:] != -100).sum())
            if counted == 0:
                continue
            total_loss += float(out.loss) * counted
            total_tokens += counted
    if total_tokens == 0:
        raise ValueError("no supervised tokens in evaluation set")
    loss = total_loss / total_tokens
    emit({
        "op": "evaluate",
        "loss": round(loss, 6),
        "perplexity": round(math.exp(min(loss, 50.0)), 6),
        "tokens": total_tokens,
        "examples": len(rows),
        "adapterLoaded": peft_info["adapterLoaded"],
        "wallClockSeconds": round(time.time() - started, 3),
        "peakRssMb": peak_rss_mb(),
        "limits": spec.get("_appliedLimits"),
    })


def op_generate(spec):
    import torch

    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    torch.manual_seed(int(spec.get("seed", 1)))
    tok, model, peft_info = build_model(spec, torch, want_training=False)
    model.eval()
    prompt = spec.get("prompt", "")
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    temperature = float(spec.get("temperature", 0.0))
    started = time.time()
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=int(spec.get("maxNewTokens", 16)),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.pad_token_id,
        )
    new_ids = out[0][ids["input_ids"].shape[1]:]
    emit({
        "op": "generate",
        "text": tok.decode(new_ids, skip_special_tokens=True),
        "newTokens": int(new_ids.shape[0]),
        "adapterLoaded": peft_info["adapterLoaded"],
        "wallClockSeconds": round(time.time() - started, 3),
        "peakRssMb": peak_rss_mb(),
        "limits": spec.get("_appliedLimits"),
    })


OPS = {"train": op_train, "evaluate": op_evaluate, "generate": op_generate}


def main():
    if len(sys.argv) < 2:
        print("usage: hf_peft_runner.py <spec.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    spec["_appliedLimits"] = apply_limits(spec)
    op = spec.get("op")
    if op not in OPS:
        print(f"unknown op: {op}", file=sys.stderr)
        return 2
    OPS[op](spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
