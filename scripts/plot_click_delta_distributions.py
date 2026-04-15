"""
Plot distributions of dx and dy from data/click_deltas.csv (run click_deltas_csv.py first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _PROJECT / "data" / "click_deltas.csv"
DEFAULT_OUTPUT = _PROJECT / "data" / "click_delta_distributions.png"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV from click_deltas_csv.py (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Histogram bins (default: 100)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    dx = df["dx"].astype(np.float64)
    dy = df["dy"].astype(np.float64)

    def describe(name: str, s: pd.Series) -> None:
        print(f"\n=== {name} (n={len(s):,}) ===")
        print(s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

    describe("dx", dx)
    describe("dy", dy)

    # Histogram range: middle bulk of data so tails do not squash the figure
    lo_x, hi_x = np.percentile(dx, [1, 99])
    lo_y, hi_y = np.percentile(dy, [1, 99])
    pad_x = 0.05 * (hi_x - lo_x + 1e-9)
    pad_y = 0.05 * (hi_y - lo_y + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dx, bins=args.bins, range=(lo_x - pad_x, hi_x + pad_x), color="C0", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Δx (consecutive clicks)")
    axes[0].set_xlabel("dx")
    axes[0].set_ylabel("count")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    axes[1].hist(dy, bins=args.bins, range=(lo_y - pad_y, hi_y + pad_y), color="C1", edgecolor="white", linewidth=0.3)
    axes[1].set_title("Δy (consecutive clicks)")
    axes[1].set_xlabel("dy")
    axes[1].set_ylabel("count")
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Histograms use 1st–99th percentile range on x-axis (full data in printed stats)")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()
