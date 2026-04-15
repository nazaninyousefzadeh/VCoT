#!/usr/bin/env python3
"""Join QA outputs with vcot_dataset_unique.json, or add QA columns to an existing merge.

Default flow: for each QA row, match ``(image, prompt)`` in the unique dataset and save
``image``, ``prompt``, ``target``, plus an answer field (default ``pointhead_answer``).

To add a second model's answers into an already-merged file (same keys), use
``--merge_into`` with the existing JSON and a new ``--qa_json`` + ``--answer_field``.

Examples::

  # Point-head answers only
  python scripts/merge_qa_pointhead_with_unique.py \\
    --qa_json runs/test_preds_two_models/point_head_2ep/qa_answers.json \\
    --answer_field pointhead_answer \\
    --out_json runs/test_preds_two_models/point_head_2ep/qa_merged_unique.json

  # Qwen LoRA v3 answers only (standalone file)
  python scripts/merge_qa_pointhead_with_unique.py \\
    --qa_json runs/test_preds_two_models/qwen_lora_v3/qa_answers.json \\
    --answer_field qwen_lora_v3_answer \\
    --out_json runs/test_preds_two_models/qwen_lora_v3/qa_merged_unique.json

  # Add qwen_lora_v3 column next to pointhead in one file
  python scripts/merge_qa_pointhead_with_unique.py \\
    --qa_json runs/test_preds_two_models/qwen_lora_v3/qa_answers.json \\
    --answer_field qwen_lora_v3_answer \\
    --merge_into runs/test_preds_two_models/point_head_2ep/qa_merged_unique.json \\
    --out_json runs/test_preds_two_models/both_qa_merged_unique.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_REPO / path).resolve()


def load_qa_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "per_sample" in data:
        return list(data["per_sample"])
    if isinstance(data, list):
        return data
    raise ValueError("QA JSON must be a list or {\"per_sample\": [...]}")


def qa_answer_map(qa_rows: list[dict]) -> dict[tuple[str, str], str]:
    """(image, prompt) -> answer string."""
    m: dict[tuple[str, str], str] = {}
    for q in qa_rows:
        img = (q.get("image") or "").strip()
        pr = (q.get("prompt") or "").strip()
        ans = q.get("answer")
        if ans is None:
            ans = q.get("response")
        m[(img, pr)] = ans
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qa_json", required=True, help="qa_answers.json from infer_qa_with_pred_clicks.py")
    p.add_argument(
        "--unique_json",
        default="data/vcot_dataset_unique.json",
        help="Full unique dataset (image, prompt, target). Used unless --merge_into is set.",
    )
    p.add_argument(
        "--answer_field",
        default="pointhead_answer",
        help="JSON key for the QA string (e.g. pointhead_answer, qwen_lora_v3_answer).",
    )
    p.add_argument(
        "--merge_into",
        default=None,
        help="Existing merged JSON list; add/update --answer_field from --qa_json per (image, prompt).",
    )
    p.add_argument("--out_json", required=True, help="Output JSON (list of objects).")
    args = p.parse_args()

    qa_path = _resolve(args.qa_json)
    out_path = _resolve(args.out_json)

    if not qa_path.is_file():
        print(f"ERROR: QA file not found: {qa_path}", file=sys.stderr)
        sys.exit(1)

    qa_rows = load_qa_rows(qa_path)
    ans_map = qa_answer_map(qa_rows)

    if args.merge_into:
        base_path = _resolve(args.merge_into)
        if not base_path.is_file():
            print(f"ERROR: --merge_into not found: {base_path}", file=sys.stderr)
            sys.exit(1)
        with open(base_path, encoding="utf-8") as f:
            merged = json.load(f)
        if not isinstance(merged, list):
            print("ERROR: --merge_into must be a JSON array", file=sys.stderr)
            sys.exit(1)
        missing = 0
        for row in merged:
            img = (row.get("image") or "").strip()
            pr = (row.get("prompt") or "").strip()
            key = (img, pr)
            if key in ans_map:
                row[args.answer_field] = ans_map[key]
            else:
                missing += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(merged)} rows -> {out_path} (added field {args.answer_field!r})")
        if missing:
            print(
                f"WARNING: {missing} base rows had no matching QA row for {args.answer_field}",
                file=sys.stderr,
            )
        return

    unique_path = _resolve(args.unique_json)
    if not unique_path.is_file():
        print(f"ERROR: unique dataset not found: {unique_path}", file=sys.stderr)
        sys.exit(1)

    with open(unique_path, encoding="utf-8") as f:
        unique_rows: list[dict] = json.load(f)

    by_key: dict[tuple[str, str], dict] = {}
    for r in unique_rows:
        img = (r.get("image") or "").strip()
        pr = (r.get("prompt") or "").strip()
        if not img or not pr:
            continue
        by_key[(img, pr)] = r

    merged: list[dict] = []
    missing: list[dict] = []

    for q in qa_rows:
        img = (q.get("image") or "").strip()
        pr = (q.get("prompt") or "").strip()
        key = (img, pr)
        base = by_key.get(key)
        if base is None:
            missing.append({"image": img, "prompt": pr, "qa_index": q.get("index")})
            continue
        row = {**base, args.answer_field: ans_map[key]}
        if "index" in q:
            row["qa_index"] = q["index"]
        merged.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} rows -> {out_path}")
    if missing:
        print(f"WARNING: {len(missing)} QA rows had no (image, prompt) match in unique dataset", file=sys.stderr)
        for m in missing[:5]:
            print(f"  unmatched: index={m.get('qa_index')} image={m['image'][:60]}...", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)


if __name__ == "__main__":
    main()
