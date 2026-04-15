#!/usr/bin/env python3
"""Run the shared test JSON through two LoRA checkpoints and save predicted clicks.

Uses ``scripts/eval_clicks_checkpoint.py`` for each adapter (same generation path as training).

Example (from repo root):

  python scripts/eval_test_set_two_models.py \\
    --test_json runs/qwen_lora_v3/test_holdout.json \\
    --out_dir runs/test_preds_two_models

Writes per model:
  - ``pred_clicks_dataset.json`` — rows ``{image, prompt, target}`` with predicted clicks
  - ``eval_full.json`` — metrics + per-sample ``pred_clicks``, ``gt_clicks``, raw prefix

Optionally ``merged_by_index.json`` — one row per test item with both models' targets.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_EVAL = _REPO / "scripts" / "eval_clicks_checkpoint.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test_json",
        default="runs/qwen_lora_v3/test_holdout.json",
        help="Test split (list of {image, prompt, target?}).",
    )
    p.add_argument(
        "--out_dir",
        default="runs/test_preds_two_models",
        help="Directory for outputs (created if missing).",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=200704,
        help="Vision resize cap (match training; passed to eval script).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max rows (debug).")
    p.add_argument("--offset", type=int, default=0, help="Skip first N rows.")
    p.add_argument(
        "--no_merged",
        action="store_true",
        help="Do not write merged_by_index.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test_json)
    if not test_path.is_absolute():
        test_path = _REPO / test_path
    if not test_path.is_file():
        print(f"ERROR: test JSON not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # name -> adapter path relative to repo
    models: list[tuple[str, str]] = [
        ("point_head_2ep", "runs/point_head_2ep/best"),
        ("qwen_lora_v3", "runs/qwen_lora_v3/best"),
    ]

    try:
        rel_test = str(test_path.relative_to(_REPO))
    except ValueError:
        rel_test = str(test_path)

    for short_name, adapter_rel in models:
        sub = out_dir / short_name
        sub.mkdir(parents=True, exist_ok=True)
        pred_ds = sub / "pred_clicks_dataset.json"
        eval_full = sub / "eval_full.json"
        try:
            rel_pred = str(pred_ds.relative_to(_REPO))
            rel_eval = str(eval_full.relative_to(_REPO))
        except ValueError:
            rel_pred = str(pred_ds)
            rel_eval = str(eval_full)
        cmd = [
            sys.executable,
            str(_EVAL),
            "--adapter",
            adapter_rel,
            "--test_json",
            rel_test,
            "--max_pixels",
            str(args.max_pixels),
            "--out_dataset_json",
            rel_pred,
            "--out_json",
            rel_eval,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.offset:
            cmd.extend(["--offset", str(args.offset)])

        print("===", short_name, "===")
        print(" ", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(_REPO))
        if r.returncode != 0:
            sys.exit(r.returncode)

    if args.no_merged:
        return

    merged_path = out_dir / "merged_by_index.json"
    rows_by_model: dict[str, list[dict]] = {}
    for short_name, _ in models:
        p = out_dir / short_name / "pred_clicks_dataset.json"
        with open(p, encoding="utf-8") as f:
            rows_by_model[short_name] = json.load(f)

    keys = list(rows_by_model.keys())
    n0 = len(rows_by_model[keys[0]])
    for k in keys[1:]:
        if len(rows_by_model[k]) != n0:
            print(
                f"WARNING: row count mismatch {keys[0]}={n0} vs {k}={len(rows_by_model[k])}; "
                "merged file may be wrong.",
                file=sys.stderr,
            )
            break

    merged: list[dict] = []
    for i in range(n0):
        base = rows_by_model[keys[0]][i]
        row = {
            "index": i + args.offset,
            "image": base.get("image"),
            "prompt": base.get("prompt"),
        }
        for short_name in keys:
            t = rows_by_model[short_name][i].get("target", "")
            row[f"target_{short_name}"] = t
        merged.append(row)

    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()
