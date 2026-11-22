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


def selected_device(torch):
    requested = os.environ.get("TMF_DEVICE", "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError(f"unsupported TMF_DEVICE={requested!r}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TMF_DEVICE=cuda requested but CUDA is unavailable")
    fraction = os.environ.get("TMF_CUDA_MEMORY_FRACTION")
    if fraction is not None:
        if requested != "cuda" or not (0 < float(fraction) <= 1):
            raise ValueError("TMF_CUDA_MEMORY_FRACTION requires CUDA and a value in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(float(fraction))
    return torch.device(requested)


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
    # A supervised completion needs an end target. Without EOS the model is
    # trained to emit the answer prefix but never to stop, while greedy exact
    # match expects the whole completion and marks every overlong answer wrong.
    if row.get("completion") and tok.eos_token_id is not None and len(full_ids) < max_len:
        full_ids.append(tok.eos_token_id)
    full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[min(len(prompt_ids), len(full_ids)):]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def build_model(spec, torch, want_training):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = spec["modelId"]
    hf = spec.get("hf") or {}
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    load_kwargs = {"local_files_only": True}
    if hf.get("loadIn4Bit"):
        from transformers import BitsAndBytesConfig

        if os.environ.get("TMF_DEVICE", "cpu") != "cuda":
            raise ValueError("4-bit model execution requires TMF_DEVICE=cuda")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=hf.get("quantType", "nf4"),
            bnb_4bit_use_double_quant=bool(hf.get("doubleQuant", True)),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = {"": 0}
    if hf.get("dtype") == "bfloat16":
        load_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if want_training and hf.get("loadIn4Bit"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(hf.get("gradientCheckpointing", False))
        )
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
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = not want_training
    return tok, model, peft_info


def place_model(model, device, spec):
    # transformers/bitsandbytes dispatches 4-bit modules during loading and
    # deliberately rejects a later .to() on quantized modules.
    if not (spec.get("hf") or {}).get("loadIn4Bit"):
        model.to(device)


def model_inputs(model, ids):
    import torch

    inputs = {"input_ids": ids}
    if getattr(model.config, "model_type", None) == "gemma4":
        inputs["mm_token_type_ids"] = torch.zeros_like(ids)
    return inputs


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

    device = selected_device(torch)
    hf = spec.get("hf") or {}
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
        use_cpu=device.type == "cpu",
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    if hf:
        args_kwargs.update(
            bf16=hf.get("dtype") == "bfloat16" and device.type == "cuda",
            gradient_checkpointing=bool(hf.get("gradientCheckpointing", False)),
            gradient_accumulation_steps=int(hf.get("gradientAccumulationSteps", 1)),
            lr_scheduler_type=hf.get("lrScheduler", "linear"),
            warmup_steps=int(hf.get("warmupSteps", 0)),
            optim=hf.get("optimizer", "adamw_torch"),
            max_grad_norm=float(hf.get("maxGradNorm", 1.0)),
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
    base_collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100)
    token_counts = {"input": 0, "supervised": 0}

    def collator(features):
        batch = base_collator(features)
        batch_tokens = int(batch["attention_mask"].sum())
        if budget.get("maxTokens") is not None and token_counts["input"] + batch_tokens > int(budget["maxTokens"]):
            raise RuntimeError("TMF_TRAIN_TOKEN_BUDGET_EXCEEDED: next batch would exceed maxTokens")
        token_counts["input"] += batch_tokens
        token_counts["supervised"] += int((batch["labels"] != -100).sum())
        if getattr(model.config, "model_type", None) == "gemma4":
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch
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
        "device": device.type,
        "peakCudaAllocatedMb": round(torch.cuda.max_memory_allocated() / 1048576, 3) if device.type == "cuda" else None,
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
        # The collator runs on each batch actually handed to Trainer, so this
        # is an attempt-level input-token count rather than whole-dataset size.
        "tokensSeen": token_counts["input"],
        "supervisedTokensSeen": token_counts["supervised"],
        "datasetTokens": int(sum(len(r["input_ids"]) for r in ds)),
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

    device = selected_device(torch)
    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    tok, model, peft_info = build_model(spec, torch, want_training=False)
    place_model(model, device, spec)
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
            ids = torch.tensor([enc["input_ids"]], device=device)
            labels = torch.tensor([enc["labels"]], device=device)
            out = model(**model_inputs(model, ids), labels=labels)
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
        "device": device.type,
        "wallClockSeconds": round(time.time() - started, 3),
        "peakRssMb": peak_rss_mb(),
        "limits": spec.get("_appliedLimits"),
    })


def op_evaluate_bundle(spec):
    """Score holdout, generate, then score validation with one model load."""
    import torch

    started = time.time()
    device = selected_device(torch)
    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    tok, model, peft_info = build_model(spec, torch, want_training=False)
    place_model(model, device, spec)
    model.eval()
    max_len = int(spec.get("maxSeqLen", 128))
    def score(path, required):
        rows = load_rows(path)
        if required and not rows:
            raise ValueError("evaluation needs at least one holdout example")
        if not rows:
            return None
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for row in rows:
                enc = encode_row(tok, row, max_len)
                if len(enc["input_ids"]) < 2:
                    continue
                ids = torch.tensor([enc["input_ids"]], device=device)
                labels = torch.tensor([enc["labels"]], device=device)
                out = model(**model_inputs(model, ids), labels=labels)
                counted = int((labels[:, 1:] != -100).sum())
                if counted == 0:
                    continue
                total_loss += float(out.loss) * counted
                total_tokens += counted
        if total_tokens == 0:
            raise ValueError("no supervised tokens in evaluation split")
        loss = total_loss / total_tokens
        return {"loss": round(loss, 6), "perplexity": round(math.exp(min(loss, 50.0)), 6),
                "tokens": total_tokens, "examples": len(rows)}

    holdout = score(spec["holdoutFile"], True)
    prompts = load_rows(spec["promptsFile"])
    if any(not isinstance(row.get("prompt"), str) for row in prompts):
        raise ValueError("evaluate_bundle prompts must be strings")
    texts = []
    with torch.no_grad():
        for row in prompts:
            torch.manual_seed(int(spec.get("seed", 1)))
            ids = tok(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
            if getattr(model.config, "model_type", None) == "gemma4":
                ids["mm_token_type_ids"] = torch.zeros_like(ids["input_ids"])
            out = model.generate(
                **ids, max_new_tokens=int(spec.get("maxNewTokens", 16)),
                do_sample=False, temperature=None, pad_token_id=tok.pad_token_id,
            )
            new_ids = out[0][ids["input_ids"].shape[1]:]
            texts.append(tok.decode(new_ids, skip_special_tokens=True))
    validation = score(spec["validationFile"], False)
    emit({
        "op": "evaluate_bundle", "holdout": holdout, "validation": validation,
        "texts": texts, "adapterLoaded": peft_info["adapterLoaded"], "device": device.type,
        "wallClockSeconds": round(time.time() - started, 3), "peakRssMb": peak_rss_mb(),
        "peakCudaAllocatedMb": round(torch.cuda.max_memory_allocated() / 1048576, 3) if device.type == "cuda" else None,
        "limits": spec.get("_appliedLimits"),
    })


def op_generate(spec):
    import torch

    device = selected_device(torch)
    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    torch.manual_seed(int(spec.get("seed", 1)))
    tok, model, peft_info = build_model(spec, torch, want_training=False)
    place_model(model, device, spec)
    model.eval()
    prompt = spec.get("prompt", "")
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if getattr(model.config, "model_type", None) == "gemma4":
        ids["mm_token_type_ids"] = torch.zeros_like(ids["input_ids"])
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
        "device": device.type,
        "wallClockSeconds": round(time.time() - started, 3),
        "peakRssMb": peak_rss_mb(),
        "limits": spec.get("_appliedLimits"),
    })


def op_generate_batch(spec):
    import torch

    device = selected_device(torch)
    torch.set_num_threads(int(spec.get("threads", 1) or 1))
    tok, model, peft_info = build_model(spec, torch, want_training=False)
    place_model(model, device, spec)
    model.eval()
    rows = load_rows(spec["promptsFile"])
    if not rows or any(not isinstance(row.get("prompt"), str) for row in rows):
        raise ValueError("generate_batch needs non-empty prompt records")
    temperature = float(spec.get("temperature", 0.0))
    max_new_tokens = int(spec.get("maxNewTokens", 16))
    texts = []
    started = time.time()
    with torch.no_grad():
        for row in rows:
            # A single-prompt subprocess reseeds on every call. Preserve that
            # behavior for sampled generation as well as greedy decoding.
            torch.manual_seed(int(spec.get("seed", 1)))
            ids = tok(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
            if getattr(model.config, "model_type", None) == "gemma4":
                ids["mm_token_type_ids"] = torch.zeros_like(ids["input_ids"])
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
            new_ids = out[0][ids["input_ids"].shape[1]:]
            texts.append(tok.decode(new_ids, skip_special_tokens=True))
    emit({
        "op": "generate_batch",
        "texts": texts,
        "count": len(texts),
        "adapterLoaded": peft_info["adapterLoaded"],
        "device": device.type,
        "wallClockSeconds": round(time.time() - started, 3),
        "peakRssMb": peak_rss_mb(),
        "peakCudaAllocatedMb": round(torch.cuda.max_memory_allocated() / 1048576, 3) if device.type == "cuda" else None,
        "limits": spec.get("_appliedLimits"),
    })


OPS = {"train": op_train, "evaluate": op_evaluate, "evaluate_bundle": op_evaluate_bundle,
       "generate": op_generate, "generate_batch": op_generate_batch}


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
