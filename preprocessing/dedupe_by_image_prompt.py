"""
Deduplicate VCoT JSON: keep one row per unique (image, prompt).

Same chart + same question may appear many times with different click trajectories
in the full dataset; this script keeps the first occurrence of each pair so each
(image, prompt) appears once. Different questions for the same image stay as
separate rows.

Run from repo root:
  python preprocessing/dedupe_by_image_prompt.py
  python preprocessing/dedupe_by_image_prompt.py -i data/vcot_dataset.json -o data/vcot_dataset_unique.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        default="data/vcot_dataset.json",
        help="Source dataset JSON (list of {image, prompt, target})",
    )
    p.add_argument(
        "-o",
        "--output",
        default="data/vcot_dataset_unique.json",
        help="Output path for deduplicated JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    in_path = root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_path = root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    if not in_path.is_file():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    with open(in_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected JSON array of samples", file=sys.stderr)
        sys.exit(1)

    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    skipped = 0

    for row in data:
        if not isinstance(row, dict):
            skipped += 1
            continue
        try:
            image = row["image"]
            prompt = row["prompt"]
        except KeyError as e:
            print(f"Warning: row missing key {e}, skipping", file=sys.stderr)
            skipped += 1
            continue

        key = (str(image), str(prompt))
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        unique.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(unique, f)

    n_in = len(data)
    n_out = len(unique)
    print(f"Read {n_in} rows from {in_path}")
    print(f"Wrote {n_out} unique (image, prompt) rows to {out_path}")
    print(f"Dropped {n_in - n_out} duplicate(s) (same image + same prompt)")


if __name__ == "__main__":
    main()
