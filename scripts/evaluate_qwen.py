"""
Evaluate inference JSON against ground_truth_answer.

Typical response text ends with: ...\\nassistant\\nThe answer is Hungary.
Matching protocol (applied in order):
  1. Numeric GT: relaxed ±5% tolerance on any number in the assistant text.
  2. Yes/No GT: last standalone yes/no word in the response must match.
  3. Exact match of normalized GT vs normalized assistant text.
  4. Order-insensitive entity match ("Iran and Pakistan" == "Pakistan and Iran").
    5. Phrase match: GT appears verbatim (word-boundary) anywhere in the response.
    6. Fuzzy match: SequenceMatcher ratio >= 0.9 on individual words (catches 1-char typos).

Usage:
  venv/bin/python scripts/evaluate_qwen.py data/qwen_responses.json
  venv/bin/python scripts/evaluate_qwen.py part1.json part2.json
  venv/bin/python scripts/evaluate_qwen.py data/out.json --examples 5

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

SalchartqaLookup = dict[tuple[str, str], tuple[str, str, bool]]

# ChartQA relaxed accuracy: |pred - gt| / |gt| <= NUMERIC_REL_TOL (gt != 0); gt == 0 uses tiny abs match.
NUMERIC_REL_TOL = 0.05
_GT_NEAR_ZERO = 1e-12

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


def numeric_relaxed_match(
    gt: float,
    pred: float,
    *,
    rel_tol: float = NUMERIC_REL_TOL,
) -> bool:
    """True if pred is within ``rel_tol`` relative error of gt (ChartQA-style)."""
    if math.isnan(gt) or math.isnan(pred) or math.isinf(gt) or math.isinf(pred):
        return False
    if abs(gt) <= _GT_NEAR_ZERO:
        return abs(pred) <= _GT_NEAR_ZERO
    return abs(pred - gt) / abs(gt) <= rel_tol


_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _word_numbers_in_text(text: str) -> list[float]:
    """Extract spelled-out numbers (one, two … twenty) from text as floats."""
    out = []
    for word in re.findall(r"\b[a-z]+\b", text.lower()):
        if word in _WORD_TO_NUM:
            out.append(float(_WORD_TO_NUM[word]))
    return out


def _token_set(s: str) -> set[str]:
    """Split 'Iran and Pakistan' -> {'iran', 'pakistan'}, ignoring conjunctions and punctuation."""
    return {
        t.strip(".,;:\"'()!?")
        for t in re.split(r"[\s,&]+", s.lower())
        if t.strip(".,;:\"'()!?") and t.strip(".,;:\"'()!?") not in ("and", "or", "the")
    }


_NO_PHRASES = [
    "does not", "do not", "did not", "is not", "are not", "was not", "were not",
    "cannot", "can not", "could not", "would not", "will not", "should not",
    "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "couldn't", "wouldn't", "won't", "shouldn't",
    "no", "incorrect", "false", "never", "neither", "none",
]
_YES_PHRASES = ["yes", "correct", "true", "affirmative", "indeed"]


def _canonical_yes_no(text: str) -> str | None:
    """Map free-form response text to 'yes', 'no', or None.

    Checks the concluding sentence first, then the full text, to avoid
    picking up negations that appear mid-reasoning but not in the answer.
    """
    t = text.lower().strip()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", t) if s.strip()]
    conclusion = sentences[-1] if sentences else t

    for scope in (conclusion, t):
        if any(p in scope for p in _NO_PHRASES):
            return "no"
        if any(p in scope for p in _YES_PHRASES):
            return "yes"

    hits = list(re.finditer(r"\b(yes|no)\b", t))
    return hits[-1].group(1) if hits else None


def _ngrams_from_text(text: str, n: int) -> list[str]:
    """Generate word n-grams from text as joined strings."""
    words = re.split(r"\s+", text.strip())
    if n == 1:
        return [w.strip(".,;:\"'()").lower() for w in words]
    return [
        " ".join(words[i : i + n]).strip(".,;:\"'()").lower()
        for i in range(len(words) - n + 1)
    ]


def _fuzzy_match(gt: str, pred: str, threshold: float = 0.85) -> bool:
    """Check if any n-gram in pred fuzzy-matches gt (handles multi-word GTs)."""
    from difflib import SequenceMatcher

    if len(gt) < 4:
        return False
    gt_words = gt.split()
    n = len(gt_words)
    for candidate in _ngrams_from_text(pred, n):
        if abs(len(candidate) - len(gt)) <= n:
            if SequenceMatcher(None, gt, candidate).ratio() >= threshold:
                return True
    return False


def match_strategy(
    prediction_raw: str | None,
    ground_truth: str | None,
    *,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> str | None:
    """
    Return the name of the first matching strategy, or None if no match.

    Strategies (applied in order):
      'numeric'   – relaxed ±numeric_rel_tol on any number in the assistant text
      'yes_no'    – last standalone yes/no word matches GT
      'exact'     – normalized GT == normalized assistant text
      'entity'    – order-insensitive token-set match
      'phrase'    – GT appears verbatim (word-boundary) inside the response
    """
    gt = normalize_ground_truth(ground_truth)
    pred = normalize_prediction(prediction_raw)
    if not gt or not pred:
        return None

    gt_num = parse_ground_truth_number(ground_truth)
    if gt_num is not None:
        for pv in numbers_in_text(pred):
            if numeric_relaxed_match(gt_num, pv, rel_tol=numeric_rel_tol):
                return "numeric"
        # word-number fallback: GT is "1" but model says "one"
        for pv in _word_numbers_in_text(pred):
            if numeric_relaxed_match(gt_num, pv, rel_tol=numeric_rel_tol):
                return "word_num"
        return None

    if gt in ("yes", "no"):
        return "yes_no" if _canonical_yes_no(pred) == gt else None

    if gt == pred:
        return "exact"

    gt_tokens = _token_set(gt)
    pred_tokens = _token_set(pred)
    if len(gt_tokens) > 1 and (gt_tokens == pred_tokens or gt_tokens.issubset(pred_tokens)):
        return "entity"

    if re.search(r"\b" + re.escape(gt) + r"\b", pred):
        return "phrase"

    # 6. fuzzy match: 1-char typos and multi-word GTs (e.g. "South Korea")
    if _fuzzy_match(gt, pred):
        return "fuzzy"

    return None


def answers_match(
    prediction_raw: str | None,
    ground_truth: str | None,
    *,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> bool:
    """True if any matching strategy fires (see match_strategy for details)."""
    return match_strategy(prediction_raw, ground_truth, numeric_rel_tol=numeric_rel_tol) is not None


def build_salchartqa_lookup(csv_path: Path) -> SalchartqaLookup:
    """
    Map (image_name, question_text) -> (image_type, question_type, is_simple).

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
            simple = (row.get("is_chart_simple") or "").strip() == "True"
            lookup[k] = (it, qt, simple)
    return lookup


def lookup_salchartqa_types(
    lookup: SalchartqaLookup, image_path: str | None, prompt: str | None
) -> tuple[str | None, str | None, bool | None]:
    """Return (image_type, question_type, is_simple) or (None, None, None) if not in CSV."""
    if not image_path:
        return None, None, None
    name = Path(image_path).name
    # saliency overlay files are named e.g. "797_Q1_overlay.png" → normalize to "797.png"
    import re as _re
    name = _re.sub(r"_Q\d+_overlay", "", name)
    q = (prompt or "").strip()
    return lookup.get((name, q), (None, None, None))


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


def accuracy(
    results: list[dict[str, Any]],
    *,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> tuple[float, int, int]:
    """Return (accuracy, n_correct, n_scored) skipping entries with no response/error."""
    correct = 0
    scored = 0
    for row in results:
        if row.get("error") or row.get("response") is None:
            continue
        scored += 1
        if answers_match(
            row.get("response"),
            row.get("ground_truth_answer"),
            numeric_rel_tol=numeric_rel_tol,
        ):
            correct += 1
    if scored == 0:
        return 0.0, 0, 0
    return correct / scored, correct, scored


def print_strategy_breakdown(
    results: list[dict[str, Any]],
    *,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> None:
    """Print how many correct answers were won by each matching strategy."""
    counts: dict[str, int] = {"numeric": 0, "word_num": 0, "yes_no": 0, "exact": 0, "entity": 0, "phrase": 0, "fuzzy": 0}
    scored = 0
    for row in results:
        if row.get("error") or row.get("response") is None:
            continue
        scored += 1
        s = match_strategy(
            row.get("response"),
            row.get("ground_truth_answer"),
            numeric_rel_tol=numeric_rel_tol,
        )
        if s:
            counts[s] = counts.get(s, 0) + 1

    total_correct = sum(counts.values())
    print("\nCorrect answers by matching strategy:")
    for name, n in counts.items():
        pct_of_correct = 100 * n / total_correct if total_correct else 0
        pct_of_scored = 100 * n / scored if scored else 0
        print(f"  {name:<8}  {n:>5}  ({pct_of_correct:.1f}% of correct | {pct_of_scored:.1f}% of scored)")


def _truncate(s: str, max_chars: int = 320) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def print_example_rows(
    results: list[dict[str, Any]],
    n_each: int,
    *,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> None:
    """Print up to ``n_each`` scored correct and incorrect rows (index, gt, assistant)."""
    if n_each <= 0:
        return
    correct_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    for row in results:
        if row.get("error") or row.get("response") is None:
            continue
        ok = answers_match(
            row.get("response"),
            row.get("ground_truth_answer"),
            numeric_rel_tol=numeric_rel_tol,
        )
        (correct_rows if ok else wrong_rows).append(row)

    print(f"\nExamples — correct (showing up to {n_each} of {len(correct_rows)}):")
    for row in correct_rows[:n_each]:
        idx = row.get("index", "?")
        gt = row.get("ground_truth_answer", "")
        ans = extract_assistant_text(row.get("response"))
        print(f"  [ok] index={idx} gt={gt!r}")
        print(f"       {_truncate(ans)}")

    print(f"\nExamples — incorrect (showing up to {n_each} of {len(wrong_rows)}):")
    for row in wrong_rows[:n_each]:
        idx = row.get("index", "?")
        gt = row.get("ground_truth_answer", "")
        ans = extract_assistant_text(row.get("response"))
        print(f"  [x]  index={idx} gt={gt!r}")
        print(f"       {_truncate(ans)}")


def _group_by_metadata(
    results: list[dict[str, Any]], lookup: SalchartqaLookup
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    int,
]:
    """Split rows into chart-type, question-type, and simplicity buckets."""
    by_chart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_simple: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched = 0
    for row in results:
        it, qt, simple = lookup_salchartqa_types(lookup, row.get("image"), row.get("prompt"))
        if it is None or qt is None:
            unmatched += 1
            continue
        by_chart[it].append(row)
        by_q[qt].append(row)
        by_simple["simple" if simple else "non-simple"].append(row)
    return dict(by_chart), dict(by_q), dict(by_simple), unmatched


def print_breakdown_by_metadata(
    results: list[dict[str, Any]],
    lookup: SalchartqaLookup,
    *,
    numeric_rel_tol: float = NUMERIC_REL_TOL,
) -> None:
    """Print accuracy per image_type, question_type, and chart complexity."""
    by_chart, by_q, by_simple, unmatched = _group_by_metadata(results, lookup)
    if unmatched:
        print(
            f"No CSV match for image+prompt (excluded from breakdown): {unmatched} rows"
        )
    if not by_chart and not by_q:
        return

    print("\nBy chart type (image_type):")
    for label in sorted(by_chart):
        acc, ok, n = accuracy(by_chart[label], numeric_rel_tol=numeric_rel_tol)
        print(f"  {label}: {acc:.4f} ({ok}/{n} scored)")

    print("\nBy question type (question_type):")
    for label in sorted(by_q):
        acc, ok, n = accuracy(by_q[label], numeric_rel_tol=numeric_rel_tol)
        print(f"  {label}: {acc:.4f} ({ok}/{n} scored)")

    print("\nBy chart complexity (is_chart_simple):")
    for label in sorted(by_simple):
        acc, ok, n = accuracy(by_simple[label], numeric_rel_tol=numeric_rel_tol)
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
    p.add_argument(
        "--numeric-rel-tol",
        type=float,
        default=NUMERIC_REL_TOL,
        metavar="F",
        help=(
            "Relaxed accuracy for numeric answers: |pred-gt|/|gt| must be <= F "
            f"(default: {NUMERIC_REL_TOL})"
        ),
    )
    p.add_argument(
        "--examples",
        type=int,
        default=0,
        metavar="N",
        help="After metrics, print N scored-correct and N scored-incorrect examples",
    )
    args = p.parse_args()

    results = merge_inference_results(args.json_paths)
    if len(args.json_paths) > 1:
        print(f"Merged {len(args.json_paths)} files -> {len(results)} rows")

    acc, n_ok, n = accuracy(results, numeric_rel_tol=args.numeric_rel_tol)
    print(f"Accuracy: {acc:.4f} ({n_ok}/{n} scored)")
    skipped = len(results) - n
    if skipped:
        print(f"Skipped (error or missing response): {skipped}")

    print_strategy_breakdown(results, numeric_rel_tol=args.numeric_rel_tol)

    if args.examples > 0:
        print_example_rows(
            results,
            args.examples,
            numeric_rel_tol=args.numeric_rel_tol,
        )

    if not args.no_metadata and args.metadata_csv.is_file():
        lookup = build_salchartqa_lookup(args.metadata_csv)
        print_breakdown_by_metadata(
            results, lookup, numeric_rel_tol=args.numeric_rel_tol
        )
    elif not args.no_metadata:
        print(f"\nMetadata CSV not found ({args.metadata_csv}), skipping type breakdown.")


if __name__ == "__main__":
    main()
