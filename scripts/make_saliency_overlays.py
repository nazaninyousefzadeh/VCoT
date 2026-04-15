"""
Pre-generate saliency overlay images for the static saliency baseline.

Blends the per-question human saliency heatmap (correct-answerer only) on top
of each chart and saves the result as a PNG. Also writes a new dataset JSON
with image paths pointing to the overlay files so inference_qwen.py can be run
without any --saliency flag.

Usage:
  venv/bin/python scripts/make_saliency_overlays.py
  venv/bin/python scripts/make_saliency_overlays.py --alpha 0.5 --output_dir data/saliency_overlays
  venv/bin/python scripts/make_saliency_overlays.py --dataset data/vcot_dataset_unique.json
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SALIENCY_DIR     = Path("data/SalChartQA/saliency_ans/heatmaps")
IMAGE_QUESTIONS  = Path("data/SalChartQA/image_questions.json")


def overlay_heatmap(chart: Image.Image, heatmap_path: Path, alpha: float) -> Image.Image:
    heatmap = Image.open(heatmap_path).convert("RGBA").resize(chart.size)
    blended = Image.blend(chart.convert("RGBA"), heatmap, alpha)
    return blended.convert("RGB")


def resolve_heatmap(img_name: str, prompt: str, image_questions: dict) -> Path | None:
    stem = Path(img_name).stem
    questions = image_questions.get(img_name, {})
    q_idx = next((k for k, v in questions.items() if v.strip() == prompt.strip()), None)
    if q_idx is None:
        return None
    path = SALIENCY_DIR / f"{stem}_{q_idx}_True.png"
    return path if path.exists() else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="data/vcot_dataset_unique.json")
    p.add_argument("--output_dir", default="data/saliency_overlays")
    p.add_argument("--output_dataset", default="data/vcot_dataset_saliency.json",
                   help="New dataset JSON pointing to overlay images")
    p.add_argument("--alpha", type=float, default=0.4,
                   help="Blend weight (0=chart only, 1=heatmap only)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        data = json.load(f)
    with open(IMAGE_QUESTIONS) as f:
        image_questions = json.load(f)

    new_dataset = []
    skipped = 0

    for sample in tqdm(data, desc="Generating overlays"):
        img_name = Path(sample["image"]).name
        heatmap_path = resolve_heatmap(img_name, sample["prompt"], image_questions)

        if heatmap_path is None:
            print(f"[warn] no heatmap for {img_name!r} / {sample['prompt'][:40]!r} — skipping")
            skipped += 1
            continue

        # output filename: {stem}_Q{n}_overlay.png
        overlay_name = heatmap_path.stem.replace("_True", "") + "_overlay.png"
        overlay_path = out_dir / overlay_name

        if not overlay_path.exists():
            chart = Image.open(sample["image"]).convert("RGB")
            blended = overlay_heatmap(chart, heatmap_path, args.alpha)
            blended.save(overlay_path)

        new_sample = dict(sample)
        new_sample["image"] = str(overlay_path)
        new_dataset.append(new_sample)

    with open(args.output_dataset, "w") as f:
        json.dump(new_dataset, f)

    print(f"\nDone — {len(new_dataset)} overlays saved to {out_dir}/")
    print(f"New dataset written to {args.output_dataset}")
    if skipped:
        print(f"Skipped (no heatmap): {skipped}")
    print(f"\nRun inference with:")
    print(f"  venv/bin/python scripts/inference_qwen.py \\")
    print(f"    --dataset {args.output_dataset} \\")
    print(f"    --output data/qwen_responses_saliency.json")


if __name__ == "__main__":
    main()
