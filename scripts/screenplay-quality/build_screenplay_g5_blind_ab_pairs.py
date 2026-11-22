#!/usr/bin/env python3
"""G5 blind A/B: anonymised pair builder and verdict tally for the 120-item suite.

Plan lines 130/137/281/284/308: two independent automatic judges score anonymised
A/B pairs; a judge never sees a model name or training stage; the candidate must
reach win rate >= 55% vs the reference with a bootstrap lower bound > 50%, and the
result must be stable when the candidate order is swapped; the answer order, seed
and judge reasons are kept.

Subcommands
  build   --suite S --reference R/outputs.jsonl --candidate C/outputs.jsonl --out-dir D [--seed N]
          Writes D/pairs_order1.jsonl and D/pairs_order2.jsonl (order2 = every pair
          swapped), each row: id, task_type, length_tier, prompt, required, forbidden,
          target_length_chars, answer_a, answer_b.  No model name in either file.
          D/key.json maps id -> {"order1": {"A": "reference"|"candidate", ...}, "order2": ...}
          D/manifest.json records inputs, sha256, seed, row counts and empty-output ids.
          Refuses id-set mismatches, missing suite ids and duplicate ids.
  tally   --key D/key.json --suite S --verdicts NAME=FILE [...]
          Each FILE: jsonl rows {"id": ..., "order": "order1"|"order2", "verdict": "A"|"B"|"tie"[, "reason": ...]}.
          NAME identifies the judge (e.g. qwen38q8, gemma26).  Prints per-judge x order
          candidate win rate (ties = 0.5), bootstrap 95% lower bound (seed 20260824,
          2000 resamples), per-task_type win rate, and order/judge agreement.  Reports
          gate ab_win_rate_vs_reference as pass/fail per judge and overall (all judges
          and both orders must pass; never promotes anything by itself).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict

GATE_WIN_RATE = 0.55
GATE_BOOT_LB = 0.50
BOOT_N = 2000
BOOT_SEED = 20260824


def sha256_16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_outputs(path: str) -> dict[str, dict]:
    rows = load_jsonl(path)
    out: dict[str, dict] = {}
    for r in rows:
        if r["id"] in out:
            sys.exit(f"duplicate id {r['id']} in {path}")
        out[r["id"]] = r
    return out


def load_suite(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    items = s["items"] if isinstance(s, dict) else s
    return {it["id"]: it for it in items}


def cmd_build(a: argparse.Namespace) -> int:
    import os

    suite = load_suite(a.suite)
    ref = load_outputs(a.reference)
    cand = load_outputs(a.candidate)
    ids_ref, ids_cand, ids_suite = set(ref), set(cand), set(suite)
    if ids_ref != ids_cand:
        sys.exit(f"id mismatch: only-reference {sorted(ids_ref - ids_cand)[:10]} only-candidate {sorted(ids_cand - ids_ref)[:10]}")
    if not ids_ref <= ids_suite:
        sys.exit(f"ids not in suite: {sorted(ids_ref - ids_suite)[:10]}")
    if a.require_full and ids_ref != ids_suite:
        sys.exit(f"outputs cover {len(ids_ref)}/{len(ids_suite)} suite ids; pass --no-require-full for a partial dry run")
    order = [it for it in suite if it in ids_ref]  # suite order, deterministic
    rng = random.Random(a.seed)
    key: dict[str, dict] = {}
    empty: dict[str, list] = {"reference": [], "candidate": []}
    rows1, rows2 = [], []
    for i in order:
        it = suite[i]
        r_txt, c_txt = ref[i].get("output", ""), cand[i].get("output", "")
        if not r_txt.strip():
            empty["reference"].append(i)
        if not c_txt.strip():
            empty["candidate"].append(i)
        cand_is_a = rng.random() < 0.5
        base = {
            "id": i,
            "task_type": it.get("task_type"),
            "length_tier": it.get("length_tier"),
            "prompt": it.get("prompt"),
            "required": it.get("required", []),
            "forbidden": it.get("forbidden", []),
            "target_length_chars": it.get("target_length_chars"),
        }
        a_txt, b_txt = (c_txt, r_txt) if cand_is_a else (r_txt, c_txt)
        rows1.append({**base, "answer_a": a_txt, "answer_b": b_txt})
        rows2.append({**base, "answer_a": b_txt, "answer_b": a_txt})
        key[i] = {
            "order1": {"A": "candidate" if cand_is_a else "reference", "B": "reference" if cand_is_a else "candidate"},
            "order2": {"A": "reference" if cand_is_a else "candidate", "B": "candidate" if cand_is_a else "reference"},
        }
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (("pairs_order1.jsonl", rows1), ("pairs_order2.jsonl", rows2)):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out_dir, "key.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "reference": a.reference_name, "candidate": a.candidate_name, "pairs": key}, f, ensure_ascii=False, indent=1)
    manifest = {
        "tool": "scripts/build_screenplay_g5_blind_ab_pairs.py build",
        "seed": a.seed,
        "suite": a.suite,
        "suite_sha256_16": sha256_16(a.suite),
        "reference": {"name": a.reference_name, "file": a.reference, "sha256_16": sha256_16(a.reference)},
        "candidate": {"name": a.candidate_name, "file": a.candidate, "sha256_16": sha256_16(a.candidate)},
        "pairs": len(order),
        "suite_items": len(suite),
        "candidate_is_A_in_order1": sum(1 for i in order if key[i]["order1"]["A"] == "candidate"),
        "empty_outputs": empty,
        "blind": "pairs_order*.jsonl carry no model name or stage; key.json must not be given to a judge",
        "gate": {"win_rate_min": GATE_WIN_RATE, "bootstrap_lower_bound_min": GATE_BOOT_LB, "both_orders_and_all_judges_must_pass": True},
    }
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("suite",)}, ensure_ascii=False))
    return 0


def score(verdict: str, a_is_cand: bool) -> float:
    if verdict == "tie":
        return 0.5
    if verdict not in ("A", "B"):
        raise ValueError(f"bad verdict {verdict!r}")
    return 1.0 if (verdict == "A") == a_is_cand else 0.0


def bootstrap_lb(vals: list[float]) -> float:
    rng = random.Random(BOOT_SEED)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(BOOT_N))
    return means[int(0.025 * BOOT_N)]


def cmd_tally(a: argparse.Namespace) -> int:
    with open(a.key, encoding="utf-8") as f:
        keyd = json.load(f)
    key = keyd["pairs"]
    suite = load_suite(a.suite)
    report: dict = {"reference": keyd["reference"], "candidate": keyd["candidate"], "judges": {}, "gate": {}}
    per_cell: dict[tuple[str, str], dict[str, float]] = {}
    for spec in a.verdicts:
        name, path = spec.split("=", 1)
        rows = load_jsonl(path)
        cells: dict[str, dict[str, float]] = defaultdict(dict)
        for r in rows:
            i, o = r["id"], r.get("order", "order1")
            if i not in key:
                sys.exit(f"{path}: id {i} not in key")
            cells[o][i] = score(r["verdict"], key[i][o]["A"] == "candidate")
        jrep = {}
        for o, sc in sorted(cells.items()):
            vals = list(sc.values())
            wr = sum(vals) / len(vals)
            lb = bootstrap_lb(vals)
            by_type: dict[str, list[float]] = defaultdict(list)
            for i, v in sc.items():
                by_type[suite[i].get("task_type", "?")].append(v)
            jrep[o] = {
                "judged": len(vals),
                "expected": len(key),
                "candidate_win_rate": round(wr, 4),
                "bootstrap95_lower": round(lb, 4),
                "ties": sum(1 for v in vals if v == 0.5),
                "pass": bool(len(vals) == len(key) and wr >= GATE_WIN_RATE and lb > GATE_BOOT_LB),
                "by_task_type": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_type.items())},
            }
            per_cell[(name, o)] = sc
        if len(jrep) == 2:
            common = set(per_cell[(name, "order1")]) & set(per_cell[(name, "order2")])
            jrep["order_agreement"] = round(sum(per_cell[(name, "order1")][i] == per_cell[(name, "order2")][i] for i in common) / max(1, len(common)), 4)
        report["judges"][name] = jrep
    names = list(report["judges"])
    if len(names) >= 2:
        agree = {}
        for o in ("order1", "order2"):
            cs = [per_cell.get((n, o)) for n in names]
            if all(cs):
                common = set.intersection(*(set(c) for c in cs))
                agree[o] = round(sum(len({c[i] for c in cs}) == 1 for i in common) / max(1, len(common)), 4)
        report["judge_agreement"] = agree
    cells_ok = [v["pass"] for j in report["judges"].values() for k, v in j.items() if k.startswith("order") and isinstance(v, dict)]
    report["gate"] = {
        "name": "ab_win_rate_vs_reference",
        "judges": len(names),
        "cells": len(cells_ok),
        "pass": bool(len(names) >= 2 and len(cells_ok) == 2 * len(names) and all(cells_ok)),
        "note": "pass requires >= 2 independent judges, both orders each, every cell win_rate >= 0.55 and bootstrap lower bound > 0.50; a pass feeds the promotion table, it is not a promotion",
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--suite", required=True)
    b.add_argument("--reference", required=True)
    b.add_argument("--candidate", required=True)
    b.add_argument("--reference-name", default="reference")
    b.add_argument("--candidate-name", default="candidate")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--seed", type=int, default=20260824)
    b.add_argument("--no-require-full", dest="require_full", action="store_false")
    b.set_defaults(func=cmd_build)
    t = sp.add_parser("tally")
    t.add_argument("--key", required=True)
    t.add_argument("--suite", required=True)
    t.add_argument("--verdicts", nargs="+", required=True, metavar="NAME=FILE")
    t.set_defaults(func=cmd_tally)
    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
