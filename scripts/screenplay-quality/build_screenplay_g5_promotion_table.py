#!/usr/bin/env python3
"""Build the G5 promotion table (plan section 8 gates) from generation-quality score files.

Inputs are score files written by score_screenplay_generation_quality.py (rules v2) for the SAME suite
and the SAME generation protocol (loader, decoder, seed, tiers). Each file may hold one or several
models. Per model, per task_type / length_tier / overall the tool records completion, cap-hit,
dup-line, rule pass, in-range and early-EOS rates and computes the plan-8 gates versus the base
row (and versus an optional DAPT rule reference). Judge-based gates (A/B win rate) and the
clean-val loss gate are external inputs and are reported as "not_computed" here, never as passed.

Verdict is "promotable_pending_external_gates" only when every computable gate passes; otherwise
"blocked" with the failing gates listed. The tool never marks anything accepted.

Usage:
  build_screenplay_g5_promotion_table.py --suite eval/screenplay_zh_eval_v1.json \
      --scores base=runs/eval120_v1/base/generation-quality-scores.json \
      --scores sft-ckpt800=runs/eval120_v1/sft-ckpt800/generation-quality-scores.json \
      [--scores multi.json]  (all models inside are taken with their own names) \
      [--base-name base] [--dapt-rule-pass 0.62] --out-json X --out-md Y
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

MAJOR_MIN_ITEMS = 10          # "主要类别": task types with at least this many suite items
SCENE_COMPLETE_MIN = 0.95     # 具体场景完整率 >= 95%
BROAD_EARLY_EOS_MAX = 0.05    # 宽泛任务提前 EOS 率 <= 5%
CATEGORY_DROP_MAX_PT = 5.0    # 任一主要类别得分不得下降超过 5 个百分点
RULE_GAIN_MIN_PT = 5.0        # 120 题硬约束命中率相对 DAPT 至少提升 5 个百分点
EARLY_EOS_FACTOR = 0.8        # rules v2 low tolerance: chars < 0.8 x target_low counts as early EOS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(vals: list[bool]) -> float | None:
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


def mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def load_models(specs: list[str]) -> "OrderedDict[str, dict]":
    models: "OrderedDict[str, dict]" = OrderedDict()
    for spec in specs:
        name_override, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        file_models = data["models"]
        if name_override and len(file_models) != 1:
            raise SystemExit(f"{spec}: NAME= override needs a single-model file, found {len(file_models)}")
        for m in file_models:
            name = name_override or m["name"]
            if name in models:
                raise SystemExit(f"duplicate model name {name}")
            models[name] = {
                "path": m.get("path"),
                "scores_file": str(p),
                "scores_sha256": sha256(p),
                "scorer_rules_version": data.get("scorer_rules_version"),
                "prompts_json": data.get("prompts_json"),
                "seed": data.get("seed"),
                "rows": {r["id"]: r for r in m["rows"]},
            }
    return models


def group_metrics(rows: list[dict], items: dict) -> dict:
    complete = [bool(r["complete"]) for r in rows]
    cap = [bool(r["hit_max_new_tokens"]) for r in rows]
    dup = [float(r["dup_line_ratio"]) for r in rows]
    constrained = [r for r in rows if r.get("has_constraints")]
    rule = [bool(r["rule_pass"]) for r in constrained]
    in_range = [r.get("length_bucket") == "in_range" for r in rows]
    early = []
    for r in rows:
        tgt = items[r["id"]].get("target_length_chars") or []
        if len(tgt) == 2 and tgt[0]:
            early.append(int(r["chars"]) < EARLY_EOS_FACTOR * float(tgt[0]))
    return {
        "n": len(rows),
        "complete_rate": rate(complete),
        "cap_hit_rate": rate(cap),
        "mean_dup_line_ratio": mean(dup),
        "constrained_items": len(constrained),
        "rule_pass_rate": rate(rule),
        "in_range_rate": rate(in_range),
        "early_eos_rate": rate(early),
        "explanation_prefix_rate": rate([bool(r.get("explanation_prefix")) for r in rows]),
        "turn_leak_rate": rate([bool(r.get("turn_leak")) for r in rows]),
        "screenplay_format_rate": rate([bool(r.get("screenplay_format")) for r in rows]),
        "mean_chars": mean([float(r["chars"]) for r in rows]),
    }


def pt(a: float | None, b: float | None) -> float | None:
    return round((a - b) * 100, 2) if a is not None and b is not None else None


def gates(cand: dict, base: dict, dapt_rule_pass: float | None, major: list[str]) -> dict:
    g: dict = {}
    # 1. rule pass gain vs DAPT (fallback: vs base, flagged)
    ref = dapt_rule_pass if dapt_rule_pass is not None else base["overall"]["rule_pass_rate"]
    gain = pt(cand["overall"]["rule_pass_rate"], ref)
    g["rule_pass_gain_pt"] = {"value": gain, "reference": "dapt" if dapt_rule_pass is not None else "base",
                              "min": RULE_GAIN_MIN_PT,
                              "status": "not_computed" if gain is None else ("pass" if gain >= RULE_GAIN_MIN_PT else "fail")}
    # 2. scene completion
    sc = cand["by_tier"].get("scene", {}).get("complete_rate")
    g["scene_complete_rate"] = {"value": sc, "min": SCENE_COMPLETE_MIN,
                                "status": "not_computed" if sc is None else ("pass" if sc >= SCENE_COMPLETE_MIN else "fail")}
    # 3. broad/long early EOS <= 5% and not worse than base
    def broad(m: dict) -> float | None:
        vals = []
        for tier in ("broad", "long"):
            t = m["by_tier"].get(tier)
            if t and t["early_eos_rate"] is not None:
                vals.extend([t["early_eos_rate"]] * t["n"])
        return round(sum(vals) / len(vals), 4) if vals else None
    be, bb = broad(cand), broad(base)
    g["broad_early_eos_rate"] = {"value": be, "base": bb, "max": BROAD_EARLY_EOS_MAX,
                                 "status": "not_computed" if be is None else ("pass" if be <= BROAD_EARLY_EOS_MAX and (bb is None or be <= bb) else "fail")}
    # 4. no major category drops > 5 pt (completion and rule pass)
    drops = {}
    for tt in major:
        c, b = cand["by_task_type"].get(tt), base["by_task_type"].get(tt)
        if not c or not b:
            continue
        d = {"complete_drop_pt": pt(b["complete_rate"], c["complete_rate"]),
             "rule_pass_drop_pt": pt(b["rule_pass_rate"], c["rule_pass_rate"])}
        d["worst_drop_pt"] = max([v for v in d.values() if v is not None], default=None)
        drops[tt] = d
    worst = max([d["worst_drop_pt"] for d in drops.values() if d["worst_drop_pt"] is not None], default=None)
    g["major_category_drop_pt"] = {"worst": worst, "max": CATEGORY_DROP_MAX_PT, "per_category": drops,
                                   "status": "not_computed" if worst is None else ("pass" if worst <= CATEGORY_DROP_MAX_PT else "fail")}
    # plan 7.2 behaviour deltas carried for the record
    g["behaviour_vs_base"] = {"completion_delta_pt": pt(cand["overall"]["complete_rate"], base["overall"]["complete_rate"]),
                              "cap_hit_delta_pt": pt(cand["overall"]["cap_hit_rate"], base["overall"]["cap_hit_rate"]),
                              "dup_line_ratio_delta": (round(cand["overall"]["mean_dup_line_ratio"] - base["overall"]["mean_dup_line_ratio"], 4)
                                                       if cand["overall"]["mean_dup_line_ratio"] is not None and base["overall"]["mean_dup_line_ratio"] is not None else None),
                              "explanation_delta_pt": pt(cand["overall"]["explanation_prefix_rate"], base["overall"]["explanation_prefix_rate"])}
    # external gates
    g["ab_win_rate_vs_reference"] = {"status": "not_computed", "min": 0.55, "note": "dual independent judges required (plan 8); supply from the judge run"}
    g["clean_val_loss_vs_dapt"] = {"status": "not_computed", "max_regression": 0.02, "note": "reference-protocol loss required"}
    failing = [k for k, v in g.items() if isinstance(v, dict) and v.get("status") == "fail"]
    pending = [k for k, v in g.items() if isinstance(v, dict) and v.get("status") == "not_computed"]
    g["verdict"] = "blocked" if failing else "promotable_pending_external_gates"
    g["failing_gates"] = failing
    g["pending_gates"] = pending
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--scores", action="append", required=True, help="[NAME=]scores.json (repeatable)")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--dapt-rule-pass", type=float, default=None, help="DAPT reference rule_pass_rate for gate 1")
    ap.add_argument("--allow-extra-rows", action="store_true",
                    help="Keep only the suite's ids from larger score files (e.g. slice quick36 out of a 120-item run); missing ids still fail.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    suite_path = Path(args.suite)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    items = {it["id"]: it for it in (suite["items"] if isinstance(suite, dict) else suite)}
    models = load_models(args.scores)
    if args.base_name not in models:
        raise SystemExit(f"base model {args.base_name!r} not among {list(models)}")
    ids = set(items)
    for name, m in models.items():
        missing = ids - set(m["rows"])
        extra = set(m["rows"]) - ids
        if extra and args.allow_extra_rows and not missing:
            m["rows"] = {i: r for i, r in m["rows"].items() if i in ids}
            m["sliced_from_rows"] = len(extra) + len(ids)
            extra = set()
        if missing or extra:
            raise SystemExit(f"{name}: rows do not match the suite (missing {len(missing)}, extra {len(extra)})")
    versions = {m["scorer_rules_version"] for m in models.values()}
    seeds = {m["seed"] for m in models.values()}
    if len(versions) != 1 or len(seeds) != 1:
        raise SystemExit(f"mixed scorer versions {versions} or seeds {seeds}; identical protocol required")

    by_tt = defaultdict(list)
    for it in items.values():
        by_tt[it["task_type"]].append(it["id"])
    major = sorted(tt for tt, v in by_tt.items() if len(v) >= MAJOR_MIN_ITEMS)

    table: "OrderedDict[str, dict]" = OrderedDict()
    for name, m in models.items():
        rows = [m["rows"][i] for i in items]
        entry = {"path": m["path"], "scores_file": m["scores_file"], "scores_sha256": m["scores_sha256"],
                 "sliced_from_rows": m.get("sliced_from_rows"),
                 "overall": group_metrics(rows, items),
                 "by_task_type": {tt: group_metrics([m["rows"][i] for i in idl], items) for tt, idl in sorted(by_tt.items())},
                 "by_tier": {}}
        tiers = defaultdict(list)
        for it in items.values():
            tiers[it.get("length_tier", "unknown")].append(it["id"])
        entry["by_tier"] = {t: group_metrics([m["rows"][i] for i in idl], items) for t, idl in sorted(tiers.items())}
        table[name] = entry
    base = table[args.base_name]
    for name, entry in table.items():
        entry["gates"] = None if name == args.base_name else gates(entry, base, args.dapt_rule_pass, major)

    out = {"tool": "build_screenplay_g5_promotion_table.py", "suite": str(suite_path), "suite_sha256": sha256(suite_path),
           "items": len(items), "scorer_rules_version": versions.pop(), "seed": seeds.pop(), "base_name": args.base_name,
           "dapt_rule_pass_reference": args.dapt_rule_pass, "major_task_types": major,
           "thresholds": {"scene_complete_min": SCENE_COMPLETE_MIN, "broad_early_eos_max": BROAD_EARLY_EOS_MAX,
                          "category_drop_max_pt": CATEGORY_DROP_MAX_PT, "rule_gain_min_pt": RULE_GAIN_MIN_PT,
                          "early_eos_factor": EARLY_EOS_FACTOR, "major_min_items": MAJOR_MIN_ITEMS},
           "models": table,
           "note": "verdicts are never 'accepted'; promotion requires the external judge and loss gates plus the loop's recorded decision"}
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# G5 promotion table ({len(items)} items, rules v{out['scorer_rules_version']}, seed {out['seed']})", "",
             "| model | complete | cap hit | dup | rule pass | in range | early EOS | scene complete | broad early EOS | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in table.items():
        o = e["overall"]; g = e["gates"]
        lines.append(f"| {name} | {o['complete_rate']} | {o['cap_hit_rate']} | {o['mean_dup_line_ratio']} | {o['rule_pass_rate']} | "
                     f"{o['in_range_rate']} | {o['early_eos_rate']} | {e['by_tier'].get('scene', {}).get('complete_rate')} | "
                     f"{g['broad_early_eos_rate']['value'] if g else '-'} | {g['verdict'] if g else 'base'} |")
    lines += ["", "## Gates (candidates vs base)", ""]
    for name, e in table.items():
        if not e["gates"]:
            continue
        g = e["gates"]
        lines.append(f"- **{name}**: {g['verdict']}; failing {g['failing_gates']}; pending {g['pending_gates']}; "
                     f"rule gain {g['rule_pass_gain_pt']['value']} pt vs {g['rule_pass_gain_pt']['reference']}; "
                     f"worst major-category drop {g['major_category_drop_pt']['worst']} pt; "
                     f"completion {g['behaviour_vs_base']['completion_delta_pt']:+} pt, cap-hit {g['behaviour_vs_base']['cap_hit_delta_pt']:+} pt, "
                     f"dup {g['behaviour_vs_base']['dup_line_ratio_delta']:+}")
    lines += ["", "## Per task type (complete / rule pass / early EOS)", "", "| task_type | n | " + " | ".join(table) + " |",
              "|---|---|" + "---|" * len(table)]
    for tt in sorted(by_tt):
        cells = []
        for e in table.values():
            m = e["by_task_type"][tt]
            cells.append(f"{m['complete_rate']} / {m['rule_pass_rate']} / {m['early_eos_rate']}")
        lines.append(f"| {tt} | {len(by_tt[tt])} | " + " | ".join(cells) + " |")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: (e["gates"]["verdict"] if e["gates"] else "base") for name, e in table.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
