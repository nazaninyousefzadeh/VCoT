"""
Write (dx, dy) for each consecutive click in fixation CSVs to a file.
First two columns are x and y (same as preprocessing/build_dataset.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = _PROJECT / "data" / "SalChartQA" / "fixationByVis"
DEFAULT_OUTPUT = _PROJECT / "data" / "click_deltas.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Consecutive click deltas from fixation CSVs.")
    p.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Folder to scan for *.csv (default: {DEFAULT_ROOT})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = p.parse_args()

    root: Path = args.root
    out: Path = args.output
    if not root.exists():
        print(f"Folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["csv_path", "step", "dx", "dy"])
        for csv_path in sorted(root.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if df.empty or df.shape[1] < 2:
                continue
            xy = df.iloc[:, :2].dropna()
            if len(xy) < 2:
                continue
            rel = csv_path.relative_to(root) if csv_path.is_relative_to(root) else csv_path
            for i in range(1, len(xy)):
                dx = float(xy.iat[i, 0]) - float(xy.iat[i - 1, 0])
                dy = float(xy.iat[i, 1]) - float(xy.iat[i - 1, 1])
                w.writerow([str(rel), i, f"{dx:.6f}", f"{dy:.6f}"])
                n += 1

    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
