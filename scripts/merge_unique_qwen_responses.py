#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_REPO / path).resolve()


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def merge_by_index(
    parts: list[tuple[str, list[dict]]],
    resume_wins: bool,
) -> tuple[list[dict], list[str]]:
    """
    Combine lists keyed by ``index``. Later parts override earlier if ``resume_wins``,
    else error on duplicate index.
    """
    by_index: dict[int, dict] = {}
    warnings: list[str] = []
    for label, rows in parts:
        for r in rows:
            idx = r.get("index")
            if not isinstance(idx, int):
                warnings.append(f"{label}: row without int index, skipped")
                continue
            if idx in by_index and not resume_wins:
                raise SystemExit(
                    f"Duplicate index {idx}: first in merge order wins; "
                    "use --resume-wins to let later files override."
                )
            if idx in by_index:
                warnings.append(f"index {idx}: overridden by {label}")
            by_index[idx] = r
    merged = [by_index[i] for i in sorted(by_index)]
    return merged, warnings


def build_key_to_global_index(vcot_rows: list[dict]) -> dict[tuple[str, str], int]:
    """``vcot_dataset_unique`` row position = global ``index`` in qwen response files."""
    m: dict[tuple[str, str], int] = {}
    for i, r in enumerate(vcot_rows):
        img = (r.get("image") or "").strip()
        pr = (r.get("prompt") or "").strip()
        k = (img, pr)
        if k in m:
            raise SystemExit(f"Duplicate (image, prompt) in unique dataset at rows {m[k]} and {i}")
        m[k] = i
    return m


def global_indices_for_test_holdout(
    test_path: Path,
    vcot_path: Path,
) -> list[int]:
    """Order matches ``test_holdout.json`` rows; values are global indices 0..5969."""
    test_rows = load_rows(test_path)
    vcot = load_rows(vcot_path)
    key_to_i = build_key_to_global_index(vcot)
    out: list[int] = []
    missing: list[str] = []
    for r in test_rows:
        k = ((r.get("image") or "").strip(), (r.get("prompt") or "").strip())
        if k not in key_to_i:
            missing.append(f"{k[0][:48]}… | {k[1][:48]}…")
        else:
            out.append(key_to_i[k])
    if missing:
        print(f"ERROR: {len(missing)} test_holdout rows not in unique dataset:", file=sys.stderr)
        for s in missing[:5]:
            print(f"  {s}", file=sys.stderr)
        raise SystemExit(1)
    return out


def align_saliency_to_indices(
    saliency_rows: list[dict],
    target_indices: list[int],
) -> tuple[list[dict | None], list[int]]:
    """For each target index, saliency row or None if missing."""
    sal_by_idx = {r["index"]: r for r in saliency_rows if isinstance(r.get("index"), int)}
    aligned: list[dict | None] = []
    missing: list[int] = []
    for idx in target_indices:
        if idx in sal_by_idx:
            aligned.append(sal_by_idx[idx])
        else:
            aligned.append(None)
            missing.append(idx)
    return aligned, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--unique",
        default="data/qwen_responses_unique.json",
        help="First chunk (e.g. indices 0..2799).",
    )
    p.add_argument(
        "--resume",
        default="data/qwen_responses_unique_resume.json",
        help="Continuation chunk (e.g. indices 2800..5969).",
    )
    p.add_argument(
        "--out-merged",
        default="data/qwen_responses_unique_merged.json",
        help="Write full merged list sorted by index.",
    )
    p.add_argument(
        "--saliency",
        default="data/qwen_responses_saliency.json",
        help="Optional: saliency-overlay inference JSON to align.",
    )
    p.add_argument(
        "--out-saliency-aligned",
        default="data/qwen_responses_saliency_aligned.json",
        help="Output: same length/order as merged; entries null or objects with error if missing.",
    )
    p.add_argument(
        "--resume-wins",
        action="store_true",
        help="If an index appears in both unique and resume, keep resume (default: error).",
    )
    p.add_argument(
        "--skip-missing-saliency-rows",
        action="store_true",
        help="Write only saliency rows that exist (shorter list); default keeps placeholders.",
    )
    p.add_argument(
        "--test-holdout",
        default=None,
        help="JSON list (image, prompt, target): write 597-row subsets matching this split.",
    )
    p.add_argument(
        "--unique-dataset",
        default="data/vcot_dataset_unique.json",
        help="Used with --test-holdout to map (image, prompt) → global index.",
    )
    p.add_argument(
        "--out-merged-test",
        default="data/qwen_responses_unique_merged_test597.json",
        help="597-row merged slice (holdout order).",
    )
    p.add_argument(
        "--out-saliency-test",
        default="data/qwen_responses_saliency_aligned_test597.json",
        help="597-row saliency slice parallel to --out-merged-test.",
    )
    args = p.parse_args()

    unique_path = _resolve(args.unique)
    resume_path = _resolve(args.resume)
    out_merged = _resolve(args.out_merged)

    if not unique_path.is_file():
        print(f"ERROR: not found: {unique_path}", file=sys.stderr)
        sys.exit(1)
    if not resume_path.is_file():
        print(f"ERROR: not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    unique_rows = load_rows(unique_path)
    resume_rows = load_rows(resume_path)

    merged, warns = merge_by_index(
        [("unique", unique_rows), ("resume", resume_rows)],
        resume_wins=args.resume_wins,
    )
    for w in warns:
        print(w, file=sys.stderr)

    out_merged.parent.mkdir(parents=True, exist_ok=True)
    with open(out_merged, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    indices = [r["index"] for r in merged if isinstance(r.get("index"), int)]
    print(f"Merged {len(merged)} rows -> {out_merged}")
    print(f"  index range: {min(indices)}..{max(indices)}")

    test_idx: list[int] | None = None
    if args.test_holdout:
        th = _resolve(args.test_holdout)
        vcot_p = _resolve(args.unique_dataset)
        if not th.is_file():
            print(f"ERROR: --test-holdout not found: {th}", file=sys.stderr)
            sys.exit(1)
        if not vcot_p.is_file():
            print(f"ERROR: --unique-dataset not found: {vcot_p}", file=sys.stderr)
            sys.exit(1)
        test_idx = global_indices_for_test_holdout(th, vcot_p)
        merged_test = [merged[i] for i in test_idx]
        out_mt = _resolve(args.out_merged_test)
        out_mt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_mt, "w", encoding="utf-8") as f:
            json.dump(merged_test, f, indent=2, ensure_ascii=False)
        print(
            f"Test holdout merged: {len(merged_test)} rows (order = {th.name}) -> {out_mt}"
        )

    sal_path = _resolve(args.saliency) if args.saliency else None
    if sal_path and sal_path.is_file():
        sal_rows = load_rows(sal_path)
        aligned, missing = align_saliency_to_indices(sal_rows, indices)
        out_sal = _resolve(args.out_saliency_aligned)

        if args.skip_missing_saliency_rows:
            out_list = [r for r in aligned if r is not None]
        else:
            by_idx_merged = {r["index"]: r for r in merged if isinstance(r.get("index"), int)}
            out_list = []
            for idx, row in zip(indices, aligned):
                if row is not None:
                    out_list.append(row)
                else:
                    base = by_idx_merged.get(idx, {})
                    out_list.append(
                        {
                            "index": idx,
                            "image": base.get("image"),
                            "prompt": base.get("prompt"),
                            "ground_truth": base.get("ground_truth"),
                            "ground_truth_clicks": base.get("ground_truth_clicks"),
                            "ground_truth_answer": base.get("ground_truth_answer"),
                            "error": "missing_in_saliency_file",
                            "response": None,
                        }
                    )

        with open(out_sal, "w", encoding="utf-8") as f:
            json.dump(out_list, f, indent=2, ensure_ascii=False)

        print(f"Saliency aligned: {len(sal_rows)} source rows -> {len(out_list)} output rows -> {out_sal}")
        if missing:
            print(f"  WARNING: saliency missing indices ({len(missing)}): {missing[:12]}{'...' if len(missing)>12 else ''}")

        if test_idx is not None:
            sal_test = [out_list[i] for i in test_idx]
            out_st = _resolve(args.out_saliency_test)
            out_st.parent.mkdir(parents=True, exist_ok=True)
            with open(out_st, "w", encoding="utf-8") as f:
                json.dump(sal_test, f, indent=2, ensure_ascii=False)
            print(f"Test holdout saliency: {len(sal_test)} rows -> {out_st}")
    elif args.saliency:
        print(f"WARNING: --saliency not found ({sal_path}), skipping aligned export.", file=sys.stderr)


if __name__ == "__main__":
    main()
