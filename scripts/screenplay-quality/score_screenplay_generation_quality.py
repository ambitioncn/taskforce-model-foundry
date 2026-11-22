#!/usr/bin/env python3
"""Offline generation-quality scorer for qualitative_screenplay_ab.py results.

Reads ``qualitative-ab-results.json`` (any number of models over the same prompt
items) plus the eval-suite JSON that carries ``required`` / ``forbidden`` /
``target_length_chars`` per item, and produces the rule-based axes from plan
sections 5 and 7.2 without a GPU:

* repetition: duplicate normalized lines, character 4-gram repeat ratio,
  longest run of identical consecutive lines;
* length: chars, cap hits (``hit_max_new_tokens``), too short / in range /
  too long versus ``target_length_chars``;
* format: scene heading present, dialogue lines, parentheticals, markdown;
* behaviour: explanation / recap / meta prefixes, turn leakage;
* completeness: ended by EOS on a sentence boundary and not truncated;
* hard constraints: all ``required`` present, no ``forbidden`` present.

Per-model aggregates and deltas versus the first model named ``base`` are
written as JSON and Markdown, together with the plan-7.2 gate flags
(rule-hit drop > 3 pt, completion drop > 5 pt, explanation rate up).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


SCENE_HEADING_RE = re.compile(
    r"^\s*(?:第?[一二三四五六七八九十百\d]+[场幕集]|场景[：:\s]|(?:内|外|内外|外内)[\.．、/／ 景]|INT\.|EXT\.|INT/EXT|"
    r"[^\n]{0,20}(?:日|夜|黄昏|清晨|傍晚|白天|夜晚|凌晨)\s*$|[^\n]{0,24}[—\-－·/／][^\n]{0,12}(?:日|夜|晨|昏)\s*$)"
)
SPEAKER_RE = re.compile(r"^\s*[一-龥A-Za-z0-9·•_]{1,10}(?:（[^）]{0,12}）|\([^)]{0,12}\))?\s*[：:]\s*\S")
PARENTHETICAL_RE = re.compile(r"[（(][^（）()]{1,20}[）)]")
# Numbered lines ("1. 内景 …") are a legitimate scene-numbering style, so only
# headings, bold, bullets and code fences count as markdown.
MARKDOWN_RE = re.compile(r"^\s*(?:#{1,6}\s|\*\*|[-*]\s|```)", re.MULTILINE)
EXPLANATION_PREFIX_RE = re.compile(
    r"^\s*(?:好的|好，|当然|没问题|以下是|下面是|这是|根据[您你]|按照[您你]|我将|我会|让我|作为|希望|注[：:]|说明[：:]|"
    r"Sure|Here is|Here's|Okay|Certainly|As an AI)"
)
META_LINE_RE = re.compile(
    r"^\s*(?:注[：:]|说明[：:]|备注[：:]|以上|希望.*(?:满意|帮助|喜欢)|如需|如果[您你]需要|我可以继续|要不要|是否需要|"
    r"（?[完结]）?\s*$|【?创作说明】?|【?写作说明】?|\*\*?(?:说明|注释|解析|分析)\*\*?[：:]?)"
)
TURN_LEAK_RE = re.compile(r"(?:<\|turn>|<turn\|>|<\|channel>|<channel\|>|\bmodel\n|^user\n|<start_of_turn>|<end_of_turn>|<\|im_start\|>|<\|im_end\|>)", re.MULTILINE)
SENTENCE_END_RE = re.compile(r"[。！？!?…”」』）)\]】—\-]\s*$|\n\s*$")
TRUNCATION_TAIL_RE = re.compile(r"[，,、：:；;（(“「『]\s*$")
# Rules v2 (G4 gate audit, 2026-09-16): trailing markdown emphasis / heading
# decoration ("**（黑屏）**", "**【切黑】**") is stripped before the tail check so
# bold-wrapped screenplay closers count as complete.
TRAILING_DECORATION_RE = re.compile(r"[*_#`\s]+$")

SHORT_LINE_MIN_CHARS = 6

# Rules v2 required-string matching: a required token matches if it appears
# literally, or after removing whitespace/punctuation from both sides
# ("货运码头-集装箱区" satisfies "货运码头集装箱区"), or - for tokens of at least
# REQUIRED_GAP_MIN_CHARS normalized characters - as an in-order character
# sequence with at most REQUIRED_GAP_PER_CHAR inserted characters between
# consecutive required characters and at most REQUIRED_GAP_TOTAL inserted in
# total ("殡仪馆家属休息室" satisfies "殡仪馆休息室"). Deletions are never
# tolerated ("便利店" does not satisfy "台风天的便利店"). Forbidden tokens match
# literally or normalized (stricter than v1).
SCORER_RULES_VERSION = 2
REQUIRED_GAP_MIN_CHARS = 5
REQUIRED_GAP_PER_CHAR = 4
REQUIRED_GAP_TOTAL = 6


def normalize_line(line: str) -> str:
    return re.sub(r"[\s\W_]+", "", line).lower()


def normalize_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text)


def required_match(token: str, text: str) -> str | None:
    """Return 'literal', 'normalized', 'gap' when ``token`` is satisfied by ``text``, else None."""
    if token in text:
        return "literal"
    key = normalize_text(token)
    body = normalize_text(text)
    if not key:
        return "literal"
    if key in body:
        return "normalized"
    if len(key) < REQUIRED_GAP_MIN_CHARS:
        return None
    pattern = "".join(
        re.escape(ch) + (f"(.{{0,{REQUIRED_GAP_PER_CHAR}}}?)" if i < len(key) - 1 else "")
        for i, ch in enumerate(key)
    )
    for match in re.finditer(pattern, body):
        if sum(len(group) for group in match.groups()) <= REQUIRED_GAP_TOTAL:
            return "gap"
    return None


def forbidden_present(token: str, text: str) -> bool:
    key = normalize_text(token)
    return token in text or (bool(key) and key in normalize_text(text))


def count_content_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def char_ngram_repeat_ratio(text: str, n: int = 4) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n + 1:
        return 0.0
    grams = [compact[i : i + n] for i in range(len(compact) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


def duplicate_line_stats(lines: list[str]) -> tuple[float, int, int]:
    """Return (duplicate ratio over long lines, duplicate count, longest identical run)."""
    normalized = [normalize_line(line) for line in lines]
    long_lines = [line for line in normalized if len(line) >= SHORT_LINE_MIN_CHARS]
    if not long_lines:
        return 0.0, 0, 0
    counts = Counter(long_lines)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    longest_run = 1
    run = 1
    for previous, current in zip(normalized, normalized[1:]):
        if current and current == previous:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
    return round(duplicates / len(long_lines), 4), duplicates, longest_run


def score_output(item: dict, output: dict) -> dict:
    text = output.get("output") or ""
    raw = output.get("raw_output") or text
    lines = [line for line in text.splitlines() if line.strip()]
    chars = count_content_chars(text)
    dup_ratio, dup_count, longest_run = duplicate_line_stats(lines)
    ngram_repeat = char_ngram_repeat_ratio(text)

    # Strip markdown emphasis / bracket decoration ("**场景：…**", "[场景：…]", "【场景 1】")
    # before matching so decorated headings are still recognised as headings.
    heading_lines = [line.strip().strip("*#[]【】 \t") for line in lines]
    scene_headings = sum(1 for line in heading_lines if line and SCENE_HEADING_RE.match(line) and len(line) <= 40)
    speaker_lines = sum(1 for line in lines if SPEAKER_RE.match(line))
    parentheticals = len(PARENTHETICAL_RE.findall(text))
    markdown_lines = len(MARKDOWN_RE.findall(text))
    explanation_prefix = bool(EXPLANATION_PREFIX_RE.match(text))
    meta_lines = sum(1 for line in lines if META_LINE_RE.match(line) and len(line.strip()) <= 60)
    turn_leak = bool(TURN_LEAK_RE.search(raw)) or bool(TURN_LEAK_RE.search(text))

    hit_cap = bool(output.get("hit_max_new_tokens"))
    tail_text = TRAILING_DECORATION_RE.sub("", text) if text.strip() else text
    tail_ok = bool(tail_text) and not TRUNCATION_TAIL_RE.search(tail_text) and bool(SENTENCE_END_RE.search(tail_text))
    complete = bool(text) and not hit_cap and tail_ok

    target = item.get("target_length_chars") or None
    length_bucket = "no_target"
    if target and len(target) == 2:
        low, high = target
        if chars < low * 0.8:
            length_bucket = "too_short"
        elif chars > high * 1.5:
            length_bucket = "too_long"
        else:
            length_bucket = "in_range"

    required = item.get("required") or []
    forbidden = item.get("forbidden") or []
    required_modes = {token: required_match(token, text) for token in required}
    required_missing = [token for token, mode in required_modes.items() if mode is None]
    required_fuzzy = {token: mode for token, mode in required_modes.items() if mode not in (None, "literal")}
    forbidden_found = [token for token in forbidden if forbidden_present(token, text)]
    has_constraints = bool(required or forbidden)
    rule_pass = not required_missing and not forbidden_found

    return {
        "id": item["id"],
        "chars": chars,
        "lines": len(lines),
        "new_tokens": output.get("new_tokens"),
        "max_new_tokens": output.get("max_new_tokens"),
        "hit_max_new_tokens": hit_cap,
        "dup_line_ratio": dup_ratio,
        "dup_line_count": dup_count,
        "longest_identical_run": longest_run,
        "char4gram_repeat_ratio": ngram_repeat,
        "scene_headings": scene_headings,
        "speaker_lines": speaker_lines,
        "parentheticals": parentheticals,
        "markdown_lines": markdown_lines,
        "explanation_prefix": explanation_prefix,
        "meta_lines": meta_lines,
        "turn_leak": turn_leak,
        "complete": complete,
        "length_bucket": length_bucket,
        "has_constraints": has_constraints,
        "required_missing": required_missing,
        "required_fuzzy": required_fuzzy,
        "forbidden_present": forbidden_found,
        "rule_pass": rule_pass,
        "screenplay_format": scene_headings >= 1 and speaker_lines >= 2,
    }


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def rate(flags: list[bool]) -> float:
    return round(sum(1 for flag in flags if flag) / len(flags), 4) if flags else 0.0


def aggregate(rows: list[dict]) -> dict:
    constrained = [row for row in rows if row["has_constraints"]]
    buckets = Counter(row["length_bucket"] for row in rows)
    return {
        "items": len(rows),
        "mean_chars": mean([row["chars"] for row in rows]),
        "median_chars": statistics.median([row["chars"] for row in rows]) if rows else 0,
        "cap_hit_rate": rate([row["hit_max_new_tokens"] for row in rows]),
        "mean_dup_line_ratio": mean([row["dup_line_ratio"] for row in rows]),
        "items_with_dup_lines": sum(1 for row in rows if row["dup_line_count"] > 0),
        "max_longest_identical_run": max((row["longest_identical_run"] for row in rows), default=0),
        "mean_char4gram_repeat_ratio": mean([row["char4gram_repeat_ratio"] for row in rows]),
        "screenplay_format_rate": rate([row["screenplay_format"] for row in rows]),
        "mean_scene_headings": mean([row["scene_headings"] for row in rows]),
        "mean_speaker_lines": mean([row["speaker_lines"] for row in rows]),
        "markdown_rate": rate([row["markdown_lines"] > 0 for row in rows]),
        "explanation_prefix_rate": rate([row["explanation_prefix"] for row in rows]),
        "meta_line_rate": rate([row["meta_lines"] > 0 for row in rows]),
        "turn_leak_rate": rate([row["turn_leak"] for row in rows]),
        "complete_rate": rate([row["complete"] for row in rows]),
        "length_buckets": dict(buckets),
        "in_range_rate": rate([row["length_bucket"] == "in_range" for row in rows if row["length_bucket"] != "no_target"]),
        "constrained_items": len(constrained),
        "rule_pass_rate": rate([row["rule_pass"] for row in constrained]),
    }


def gate_flags(base: dict, candidate: dict) -> dict:
    """Plan 7.2 DAPT promotion gates relative to base (percentage points)."""
    rule_drop_pt = round((base["rule_pass_rate"] - candidate["rule_pass_rate"]) * 100, 2) if base["constrained_items"] else None
    completion_drop_pt = round((base["complete_rate"] - candidate["complete_rate"]) * 100, 2)
    explanation_up_pt = round((candidate["explanation_prefix_rate"] + candidate["meta_line_rate"] - base["explanation_prefix_rate"] - base["meta_line_rate"]) * 100, 2)
    repetition_up = round(candidate["mean_dup_line_ratio"] - base["mean_dup_line_ratio"], 4)
    return {
        "rule_hit_drop_pt": rule_drop_pt,
        "rule_gate_stop": bool(rule_drop_pt is not None and rule_drop_pt > 3),
        "completion_drop_pt": completion_drop_pt,
        "completion_gate_block": completion_drop_pt > 5,
        "explanation_recap_up_pt": explanation_up_pt,
        "explanation_gate_block": explanation_up_pt > 5,
        "dup_line_ratio_delta": repetition_up,
        "cap_hit_rate_delta_pt": round((candidate["cap_hit_rate"] - base["cap_hit_rate"]) * 100, 2),
        "format_rate_delta_pt": round((candidate["screenplay_format_rate"] - base["screenplay_format_rate"]) * 100, 2),
        "turn_leak_rate_delta_pt": round((candidate["turn_leak_rate"] - base["turn_leak_rate"]) * 100, 2),
    }


def load_suite_items(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    return {item["id"]: item for item in items}


def score_results(results: dict, suite_items: dict[str, dict] | None) -> dict:
    prompt_items = {item["id"]: dict(item) for item in results["prompts"]}
    if suite_items:
        for item_id, item in prompt_items.items():
            if item_id in suite_items:
                for key in ("required", "forbidden", "target_length_chars", "task_type", "length_tier"):
                    if key in suite_items[item_id]:
                        item[key] = suite_items[item_id][key]
    scored_models = []
    for model in results["models"]:
        rows = []
        for output in model["outputs"]:
            item = prompt_items.get(output["id"])
            if item is None:
                raise ValueError(f"output id {output['id']!r} for model {model['name']} is not in the prompt list")
            rows.append(score_output(item, output))
        scored_models.append({"name": model["name"], "path": model.get("path"), "aggregate": aggregate(rows), "rows": rows})

    base = next((model for model in scored_models if model["name"] == "base"), None)
    for model in scored_models:
        model["gates_vs_base"] = gate_flags(base["aggregate"], model["aggregate"]) if base and model is not base else None
    return {
        "scorer_rules_version": SCORER_RULES_VERSION,
        "source_base_model": results.get("base_model"),
        "prompts_json": results.get("prompts_json"),
        "seed": results.get("seed"),
        "models": scored_models,
    }


AGG_COLUMNS = [
    ("mean_chars", "chars"),
    ("cap_hit_rate", "cap hit"),
    ("complete_rate", "complete"),
    ("mean_dup_line_ratio", "dup lines"),
    ("max_longest_identical_run", "max run"),
    ("mean_char4gram_repeat_ratio", "4gram rep"),
    ("screenplay_format_rate", "format"),
    ("explanation_prefix_rate", "explain"),
    ("meta_line_rate", "meta"),
    ("markdown_rate", "markdown"),
    ("turn_leak_rate", "leak"),
    ("in_range_rate", "len ok"),
    ("rule_pass_rate", "rule pass"),
]


def render_markdown(report: dict) -> str:
    lines = ["# Screenplay generation-quality scores", ""]
    lines.append(f"- Base model: `{report.get('source_base_model')}`")
    lines.append(f"- Prompts: `{report.get('prompts_json')}`; seed {report.get('seed')}")
    lines.append("")
    header = "| model | n | " + " | ".join(label for _, label in AGG_COLUMNS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(AGG_COLUMNS) + 2))
    for model in report["models"]:
        agg = model["aggregate"]
        cells = [str(agg[key]) for key, _ in AGG_COLUMNS]
        lines.append(f"| {model['name']} | {agg['items']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Plan 7.2 gates versus base (percentage points)")
    lines.append("")
    lines.append("| model | rule drop | rule stop | completion drop | completion block | explain/recap up | explain block | dup delta | cap delta | format delta | leak delta |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in report["models"]:
        gates = model.get("gates_vs_base")
        if not gates:
            continue
        lines.append(
            f"| {model['name']} | {gates['rule_hit_drop_pt']} | {gates['rule_gate_stop']} | {gates['completion_drop_pt']} | "
            f"{gates['completion_gate_block']} | {gates['explanation_recap_up_pt']} | {gates['explanation_gate_block']} | "
            f"{gates['dup_line_ratio_delta']} | {gates['cap_hit_rate_delta_pt']} | {gates['format_rate_delta_pt']} | {gates['turn_leak_rate_delta_pt']} |"
        )
    lines.append("")
    lines.append("## Per-item flags")
    lines.append("")
    lines.append("| item | model | chars | cap | complete | dup | run | format | explain | meta | leak | len | rule |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in report["models"]:
        for row in model["rows"]:
            rule = "-" if not row["has_constraints"] else ("pass" if row["rule_pass"] else f"miss={row['required_missing']} forb={row['forbidden_present']}")
            lines.append(
                f"| {row['id']} | {model['name']} | {row['chars']} | {int(row['hit_max_new_tokens'])} | {int(row['complete'])} | "
                f"{row['dup_line_ratio']} | {row['longest_identical_run']} | {int(row['screenplay_format'])} | {int(row['explanation_prefix'])} | "
                f"{row['meta_lines']} | {int(row['turn_leak'])} | {row['length_bucket']} | {rule} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="qualitative-ab-results.json from qualitative_screenplay_ab.py")
    parser.add_argument("--suite", default="", help="Eval suite JSON with required/forbidden/target_length_chars per item")
    parser.add_argument("--out-json", default="", help="Default: <results dir>/generation-quality-scores.json")
    parser.add_argument("--out-md", default="", help="Default: <results dir>/generation-quality-scores.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = Path(args.results)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    suite_items = load_suite_items(Path(args.suite)) if args.suite else None
    report = score_results(results, suite_items)
    out_json = Path(args.out_json) if args.out_json else results_path.parent / "generation-quality-scores.json"
    out_md = Path(args.out_md) if args.out_md else results_path.parent / "generation-quality-scores.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    for model in report["models"]:
        agg = model["aggregate"]
        print(
            f"{model['name']}: n={agg['items']} chars={agg['mean_chars']} cap_hit={agg['cap_hit_rate']} complete={agg['complete_rate']} "
            f"dup={agg['mean_dup_line_ratio']} format={agg['screenplay_format_rate']} explain={agg['explanation_prefix_rate']} "
            f"leak={agg['turn_leak_rate']} rule_pass={agg['rule_pass_rate']}"
        )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
