"""Draw click-sequence arrows on an image (0–1000 normalized coords by default)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

# Same separator as preprocessing/vcot_target.TARGET_SEP
_TARGET_SEP = " | <sep> | "
_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")


def parse_clicks_from_target(target: str) -> list[tuple[float, float]]:
    """Extract ``(x, y)`` clicks from a dataset ``target`` string (0–1000 normalized)."""
    if _TARGET_SEP in target:
        clicks_part = target.split(_TARGET_SEP, 1)[0].strip()
    else:
        clicks_part = target.strip()
    return [(float(x), float(y)) for x, y in _RE_CLICK.findall(clicks_part)]


def load_dataset_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load a JSON dataset: top-level list of objects with ``image`` and ``target``."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {p}, got {type(data).__name__}")
    return data


def resolve_image_path(image_field: str, repo_root: Path) -> Path:
    """Resolve ``image`` field (often repo-relative) to an absolute path."""
    img = Path(image_field)
    return img.resolve() if img.is_absolute() else (repo_root / img).resolve()


def plot_click_arrows_from_dataset_row(
    dataset_json: Path | str,
    index: int,
    *,
    repo_root: Path | None = None,
    **plot_kw: Any,
) -> tuple[Figure, Axes, list[tuple[float, float]]]:
    """
    Load row ``index`` from ``dataset_json``, parse clicks from ``target``, open ``image``,
    and draw arrows. Paths in JSON are resolved relative to ``repo_root`` (project root).

    Returns ``(fig, ax, clicks)`` with clicks in 0–1000 normalized form.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    rows = load_dataset_rows(dataset_json)
    if index < 0 or index >= len(rows):
        raise IndexError(f"index {index} out of range (len={len(rows)})")
    row = rows[index]
    clicks = parse_clicks_from_target(row["target"])
    img_path = resolve_image_path(row["image"], repo_root)
    pil = Image.open(img_path).convert("RGB")
    fig, ax = plot_click_arrows(pil, clicks, **plot_kw)
    return fig, ax, clicks


def plot_compare_from_dataset_rows(
    pred_json: Path | str,
    gt_json: Path | str,
    index: int,
    *,
    repo_root: Path | None = None,
) -> tuple[Figure, Axes, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Overlay predicted vs ground-truth clicks for the same row index in two dataset JSONs.
    Image path is taken from the **prediction** row.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    rows_p = load_dataset_rows(pred_json)
    rows_g = load_dataset_rows(gt_json)
    if index < 0 or index >= len(rows_p):
        raise IndexError(f"pred index {index} out of range (len={len(rows_p)})")
    if index >= len(rows_g):
        raise IndexError(f"gt index {index} out of range (len={len(rows_g)})")
    pred = parse_clicks_from_target(rows_p[index]["target"])
    gt = parse_clicks_from_target(rows_g[index]["target"])
    img_path = resolve_image_path(rows_p[index]["image"], repo_root)
    pil = Image.open(img_path).convert("RGB")
    fig, ax = plot_click_arrows_compare(pil, pred, gt)
    return fig, ax, pred, gt


def _to_pil_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return image.convert("RGB")


def clicks_to_pixels(
    clicks: Sequence[tuple[float, float]],
    width: int,
    height: int,
    *,
    normalized: bool = True,
    coord_scale: float = 1000.0,
) -> list[tuple[float, float]]:
    """Map click coordinates to pixel space for overlay on ``width`` x ``height``."""
    out: list[tuple[float, float]] = []
    for x, y in clicks:
        if normalized:
            px = x / coord_scale * width
            py = y / coord_scale * height
        else:
            px, py = float(x), float(y)
        out.append((px, py))
    return out


def _draw_click_sequence_pixels(
    ax: Axes,
    pts: list[tuple[float, float]],
    *,
    arrow_color: str,
    point_color: str,
    point_edge: str,
    number_color: str,
    linewidth: float,
    markersize: float,
    number_clicks: bool,
    z_arrow: int,
    z_points: int,
    z_text: int,
) -> None:
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=linewidth,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=z_arrow,
        )

    if pts:
        xs, ys = zip(*pts)
        ax.scatter(
            xs,
            ys,
            c=point_color,
            s=markersize**2 * 1.8,
            edgecolors=point_edge,
            linewidths=1.0,
            zorder=z_points,
        )
        if number_clicks:
            for i, (px, py) in enumerate(pts, start=1):
                ax.annotate(
                    str(i),
                    (px, py),
                    color=number_color,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=z_text,
                )


def plot_click_arrows(
    image: Image.Image | np.ndarray,
    clicks: Sequence[tuple[float, float]],
    *,
    normalized: bool = True,
    coord_scale: float = 1000.0,
    ax: Axes | None = None,
    arrow_color: str = "#00ff88",
    point_color: str = "#ffffff",
    point_edge: str = "#222222",
    linewidth: float = 2.0,
    markersize: float = 8.0,
    number_clicks: bool = True,
) -> tuple[Figure, Axes]:

    pil = _to_pil_rgb(image)
    w, h = pil.size
    pts = clicks_to_pixels(list(clicks), w, h, normalized=normalized, coord_scale=coord_scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.imshow(pil, origin="upper")
    ax.set_axis_off()
    _draw_click_sequence_pixels(
        ax,
        pts,
        arrow_color=arrow_color,
        point_color=point_color,
        point_edge=point_edge,
        number_color="black",
        linewidth=linewidth,
        markersize=markersize,
        number_clicks=number_clicks,
        z_arrow=3,
        z_points=5,
        z_text=6,
    )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.tight_layout()
    return fig, ax


def plot_click_arrows_compare(
    image: Image.Image | np.ndarray,
    pred: Sequence[tuple[float, float]],
    gt: Sequence[tuple[float, float]],
    *,
    normalized: bool = True,
    coord_scale: float = 1000.0,
    ax: Axes | None = None,
    linewidth: float = 2.0,
    markersize: float = 8.0,
    number_clicks: bool = True,
) -> tuple[Figure, Axes]:
    """Overlay predicted (cyan) and ground-truth (orange) click sequences on one figure."""
    pil = _to_pil_rgb(image)
    w, h = pil.size
    pts_p = clicks_to_pixels(list(pred), w, h, normalized=normalized, coord_scale=coord_scale)
    pts_g = clicks_to_pixels(list(gt), w, h, normalized=normalized, coord_scale=coord_scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.imshow(pil, origin="upper")
    ax.set_axis_off()

    # GT drawn first (under); prediction on top
    _draw_click_sequence_pixels(
        ax,
        pts_g,
        arrow_color="#ff8800",
        point_color="#ffe0b0",
        point_edge="#aa4400",
        number_color="#663300",
        linewidth=linewidth,
        markersize=markersize * 0.95,
        number_clicks=number_clicks,
        z_arrow=3,
        z_points=5,
        z_text=6,
    )
    _draw_click_sequence_pixels(
        ax,
        pts_p,
        arrow_color="#00b4d8",
        point_color="#e0f7ff",
        point_edge="#0077b6",
        number_color="#003049",
        linewidth=linewidth,
        markersize=markersize,
        number_clicks=number_clicks,
        z_arrow=4,
        z_points=7,
        z_text=8,
    )

    ax.legend(
        handles=[
            Line2D([0], [0], color="#ff8800", lw=2, label="Ground truth"),
            Line2D([0], [0], color="#00b4d8", lw=2, label="Predicted"),
        ],
        loc="lower right",
        framealpha=0.92,
    )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.tight_layout()
    return fig, ax


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay click arrows on an image.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="VCoT-style JSON (list of {image, prompt, target}); clicks from <click>x,y</click> in target.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index when using --dataset (default: 0). Ignored if --indices is set.",
    )
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated row indices for --dataset (e.g. 2,3,4). Saves one PNG per row: "
        "{output_stem}_{idx}{suffix}. Overrides --index.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Plot every row in --dataset (saves one PNG per row). Incompatible with --index/--indices.",
    )
    p.add_argument(
        "--gt-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional second JSON (e.g. test_holdout.json) with ground-truth targets. "
        "Same row index as --dataset; overlays GT (orange) vs pred (cyan).",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Resolve relative image paths against this dir (default: VCoT project root).",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to image (PNG/JPEG). If omitted, use --demo.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use a solid placeholder image (no --image needed).",
    )
    p.add_argument(
        "--click",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="One click (x y). Repeat for each point. Not used with --dataset.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save PNG here (default: click_arrows.png in cwd).",
    )
    p.add_argument(
        "--pixels",
        action="store_true",
        help="Interpret clicks as pixel coordinates instead of 0–1000 normalized.",
    )
    args = p.parse_args()

    if args.gt_dataset is not None and args.dataset is None:
        p.error("--gt-dataset requires --dataset (predictions JSON).")

    out = args.output or Path("click_arrows.png")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset is not None:
        repo_root = args.repo_root
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]
        if args.pixels:
            p.error("--pixels is only for manual --click mode; dataset targets are always 0–1000.")
        if args.all and args.indices is not None:
            p.error("Use either --all or --indices, not both.")
        if args.all:
            idx_list = list(range(len(load_dataset_rows(args.dataset))))
            if not idx_list:
                p.error("Dataset is empty.")
        elif args.indices:
            idx_list = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        else:
            idx_list = [args.index]
        for idx in idx_list:
            if args.gt_dataset is not None:
                fig, _, n_pred, n_gt = plot_compare_from_dataset_rows(
                    args.dataset,
                    args.gt_dataset,
                    idx,
                    repo_root=repo_root,
                )
                msg_clicks = f"pred={len(n_pred)} gt={len(n_gt)}"
            else:
                fig, _, n_clicks = plot_click_arrows_from_dataset_row(
                    args.dataset,
                    idx,
                    repo_root=repo_root,
                    normalized=True,
                )
                msg_clicks = f"{len(n_clicks)} clicks"
            if len(idx_list) > 1:
                save_path = out.parent / f"{out.stem}_{idx}{out.suffix}"
            else:
                save_path = out
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {save_path} (dataset row {idx}, {msg_clicks})")
        return

    if args.demo and args.image is not None:
        p.error("Use either --demo or --image, not both.")
    if not args.demo and args.image is None:
        p.error("Provide --dataset, or (--image PATH | --demo) with --click.")
    if not args.click:
        p.error("--click is required unless --dataset is used.")

    if args.demo:
        img = Image.new("RGB", (640, 480), color=(35, 45, 70))
    else:
        img = Image.open(args.image).convert("RGB")

    clicks = [(float(a), float(b)) for a, b in args.click]
    fig, _ = plot_click_arrows(img, clicks, normalized=not args.pixels)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
