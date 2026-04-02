"""
Evaluate inference JSON against ground_truth_answer.

Typical response text ends with: ...\\nassistant\\nThe answer is Hungary.
Matching: numeric ground-truth labels compare to any number in the assistant text
(42 vs 42.0); otherwise whole-word containment / exact match on normalized text.

Usage:
  venv/bin/python scripts/evaluate_qwen.py data/qwen_responses.json
  venv/bin/python scripts/evaluate_qwen.py part1.json part2.json

  With multiple files, rows are merged by ``index``; later files overwrite
  earlier ones for the same index (pass partial run first, resume second).

  With ``--metadata-csv`` (SalChartQA ``unified_approved.csv``), also prints
  accuracy by chart type (``image_type``) and question type (``question_type``),
  matching each row on image file name + prompt text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SalchartqaLookup = dict[tuple[str, str], tuple[str, str]]

# Integers, decimals, scientific notation, optional thousands separators (e.g. 1,234.5).
_NUM_IN_TEXT = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
# Short label that is only a number (optional commas, trailing %).
_NUM_LABEL = re.compile(
    r"^\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?\s*$"
)


ASSISTANT_MARKERS = ("\nassistant\n", "\nassistant")


def extract_assistant_text(raw: str | None) -> str:
    """Return the model turn only (text after the last assistant marker)."""
    if not raw or not isinstance(raw, str):
        return ""
    text = raw
    for m in ASSISTANT_MARKERS:
        if m in text:
            text = text.rsplit(m, 1)[-1]
    return text.strip()


def normalize_label(s: str | None) -> str:
    """Normalize a short ground-truth or candidate answer string."""
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_prediction(raw: str | None) -> str:
    """Clean model output: assistant slice only, then same normalization as labels."""
    return normalize_label(extract_assistant_text(raw))


def normalize_ground_truth(gt: str | None) -> str:
    """Normalize dataset ground_truth_answer (same rules as prediction)."""
    return normalize_label(gt)


def parse_ground_truth_number(gt: str | None) -> float | None:
    """
    If ground_truth_answer is a single numeric value (optional commas or trailing
    %), return it as float; otherwise None.
    """
    if gt is None:
        return None
    raw = str(gt).strip()
    if not raw or not _NUM_LABEL.match(raw):
        return None
    t = raw.replace(",", "").rstrip()
    if t.endswith("%"):
        t = t[:-1].strip()
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in_text(s: str) -> list[float]:
    """Extract numeric literals from a string for comparison to a numeric label."""
    out: list[float] = []
    for m in _NUM_IN_TEXT.finditer(s):
        token = m.group(0).replace(",", "")
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def numbers_close(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> bool:
    """True if two floats match for evaluation (handles 42 vs 42.0)."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def answers_match(prediction_raw: str | None, ground_truth: str | None) -> bool:
    """
    True if the normalized label matches the prediction.

    If ground truth parses as a number, any matching number in the assistant
    text counts (e.g. 42 vs "The value is 42.0").

    Otherwise uses whole-word containment so a one-word ground truth can match
    inside a full sentence, plus exact string equality after normalization.
    """
    gt = normalize_ground_truth(ground_truth)
    pred = normalize_prediction(prediction_raw)
    if not gt:
        return False
    if not pred:
        return False

    gt_num = parse_ground_truth_number(ground_truth)
    if gt_num is not None:
        for pv in numbers_in_text(pred):
            if numbers_close(gt_num, pv):
                return True
        return False

    if gt == pred:
        return True
    if re.search(rf"(?<!\w){re.escape(gt)}(?!\w)", pred):
        return True
    return False

def compute_error(prediction_raw: str | None, ground_truth: str | None) -> dict:
    """
    Returns a dict with:
    - type: "numeric" or "text" or "none"
    - value: error metric (float or int)
    """
    pred = normalize_prediction(prediction_raw)
    gt = normalize_ground_truth(ground_truth)

    if not pred or not gt:
        return {"type": "none", "value": None}

    # --- NUMERIC CASE ---
    gt_num = parse_ground_truth_number(ground_truth)
    if gt_num is not None:
        pred_numbers = numbers_in_text(pred)
        if not pred_numbers:
            return {"type": "numeric", "value": None}

        # take closest number in prediction
        closest = min(pred_numbers, key=lambda x: abs(x - gt_num))

        if gt_num == 0:
            error = abs(closest - gt_num)
        else:
            error = abs(closest - gt_num) / abs(gt_num) * 100

        return {"type": "numeric", "value": error}

    # --- TEXT CASE (edit distance) ---
    def levenshtein(a: str, b: str) -> int:
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1):
            dp[i][0] = i
        for j in range(len(b)+1):
            dp[0][j] = j

        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # delete
                    dp[i][j-1] + 1,      # insert
                    dp[i-1][j-1] + cost  # substitute
                )
        return dp[-1][-1]

    dist = levenshtein(pred, gt)/len(gt)

    return {"type": "text", "value": dist}

def build_salchartqa_lookup(csv_path: Path) -> SalchartqaLookup:
    """
    Map (image_name, question_text) -> (image_type, question_type).

    Duplicate rows in the CSV share the same types; first row wins.
    """
    lookup: SalchartqaLookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = (row.get("image_name") or "").strip()
            q = (row.get("question") or "").strip()
            if not img or not q:
                continue
            k = (img, q)
            if k in lookup:
                continue
            it = (row.get("image_type") or "").strip() or "unknown"
            qt = (row.get("question_type") or "").strip() or "unknown"
            lookup[k] = (it, qt)
    return lookup


def lookup_salchartqa_types(
    lookup: SalchartqaLookup, image_path: str | None, prompt: str | None
) -> tuple[str | None, str | None]:
    """Return (image_type, question_type) or (None, None) if not in CSV."""
    if not image_path:
        return None, None
    name = Path(image_path).name
    q = (prompt or "").strip()
    return lookup.get((name, q), (None, None))


def merge_inference_results(paths: list[Path]) -> list[dict[str, Any]]:
    """
    Load several inference JSON lists and combine them.

    Rows with an integer ``index`` are keyed by index; the last occurrence wins
    (useful when a stopped run and a resume both contain overlapping indices).

    Rows without ``index`` are appended at the end in file order.
    """
    by_index: dict[int, dict[str, Any]] = {}
    tail_no_index: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as f:
            batch = json.load(f)
        for row in batch:
            idx = row.get("index")
            if isinstance(idx, int):
                by_index[idx] = row
            else:
                tail_no_index.append(row)
    merged = [by_index[i] for i in sorted(by_index)]
    merged.extend(tail_no_index)
    return merged


def accuracy(results: list[dict[str, Any]]) -> tuple[float, int, int]:
    """
    Return (accuracy, n_correct, n_scored) skipping entries with no response/error.
    """
    correct = 0
    scored = 0
    for row in results:
        if row.get("error") or row.get("response") is None:
            continue
        scored += 1
        if answers_match(row.get("response"), row.get("ground_truth_answer")):
            correct += 1
        else:
            row["computed_error"] = compute_error(row.get("response"),row.get("ground_truth_answer"))
    if scored == 0:
        return 0.0, 0, 0
    return correct / scored, correct, scored


def _group_by_metadata(
    results: list[dict[str, Any]], lookup: SalchartqaLookup
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], int]:
    """Split rows into chart-type and question-type buckets; return (by_chart, by_q, n_unmatched)."""
    by_chart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched = 0
    for row in results:
        it, qt = lookup_salchartqa_types(lookup, row.get("image"), row.get("prompt"))
        if it is None or qt is None:
            unmatched += 1
            continue
        by_chart[it].append(row)
        by_q[qt].append(row)
    return dict(by_chart), dict(by_q), unmatched


def print_breakdown_by_metadata(results: list[dict[str, Any]], lookup: SalchartqaLookup) -> None:
    """Print accuracy per image_type and per question_type (SalChartQA labels)."""
    by_chart, by_q, unmatched = _group_by_metadata(results, lookup)
    if unmatched:
        print(
            f"No CSV match for image+prompt (excluded from breakdown): {unmatched} rows"
        )
    if not by_chart and not by_q:
        return

    print("\nBy chart type (image_type):")
    for label in sorted(by_chart):
        acc, ok, n = accuracy(by_chart[label])
        print(f"  {label}: {acc:.4f} ({ok}/{n} scored)")

    print("\nBy question type (question_type):")
    for label in sorted(by_q):
        acc, ok, n = accuracy(by_q[label])
        print(f"  {label}: {acc:.4f} ({ok}/{n} scored)")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute accuracy on inference JSON.")
    p.add_argument(
        "json_paths",
        nargs="+",
        type=Path,
        help="One or more qwen_responses*.json files (merged by index, last wins)",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/SalChartQA/unified_approved.csv"),
        help="SalChartQA unified_approved.csv for chart/question type breakdown",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not load CSV; only print overall accuracy",
    )
    args = p.parse_args()

    results = merge_inference_results(args.json_paths)
    if len(args.json_paths) > 1:
        print(f"Merged {len(args.json_paths)} files -> {len(results)} rows")

    acc, n_ok, n = accuracy(results)
    print(f"Accuracy: {acc:.4f} ({n_ok}/{n} scored)")
    skipped = len(results) - n
    if skipped:
        print(f"Skipped (error or missing response): {skipped}")

    if not args.no_metadata and args.metadata_csv.is_file():
        lookup = build_salchartqa_lookup(args.metadata_csv)
        print_breakdown_by_metadata(results, lookup)
    elif not args.no_metadata:
        print(f"\nMetadata CSV not found ({args.metadata_csv}), skipping type breakdown.")


if __name__ == "__main__":
    main()
