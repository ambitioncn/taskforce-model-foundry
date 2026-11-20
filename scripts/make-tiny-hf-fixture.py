#!/usr/bin/env python3
"""Build a tiny, randomly initialised local HF model + tokenizer fixture.

Nothing is downloaded and no pretrained weights are used: the tokenizer is a
word-level vocabulary built from a fixed synthetic word list and the model is a
2-layer Llama with random init. This exists so the hf-peft backend can be
exercised end to end on CPU without network access, accounts or paid compute.

usage: make-tiny-hf-fixture.py <outDir> [<dataOutJsonl>]
"""
import json
import os
import sys

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "one", "two", "three", "four", "five", "six", "seven", "eight",
         "map", "to", "is", "value", "key", "code", "next", "prev"]
SPECIALS = ["<unk>", "<pad>", "<s>", "</s>"]


def build_tokenizer(out_dir):
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>",
    )
    fast.save_pretrained(out_dir)
    return fast, len(vocab)


def build_model(out_dir, vocab_size):
    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(out_dir, safe_serialization=True)
    return sum(p.numel() for p in model.parameters())


def build_data(path, n=48):
    """Deterministic synthetic prompt/completion pairs with a learnable rule."""
    pairs = [("alpha", "one"), ("beta", "two"), ("gamma", "three"), ("delta", "four")]
    rows = []
    for i in range(n):
        src, dst = pairs[i % len(pairs)]
        rows.append({"prompt": f"map {src} to", "completion": f" {dst}"})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    _, vocab_size = build_tokenizer(out_dir)
    params = build_model(out_dir, vocab_size)
    result = {"modelDir": out_dir, "vocabSize": vocab_size, "params": params}
    if len(sys.argv) > 2:
        result["dataFile"] = sys.argv[2]
        result["rows"] = build_data(sys.argv[2])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
