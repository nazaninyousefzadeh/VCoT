#!/usr/bin/env python3
"""Run inference with a saved LoRA adapter (e.g. runs/qwen_lora_v3/best).

Defaults use mild sampling + repetition_penalty to reduce greedy collapse (same coordinate
repeated many times). Use --greedy to match train_updated.py generation_eval (often worse).

Important: Qwen2-VL's hub ``generation_config`` sets ``top_k=1``. If you pass ``temperature`` /
``top_p`` but not ``top_k``, sampling is still effectively greedy (one token per step). This
script sets ``top_k=0`` when sampling (disables top-k filtering; ``top_p`` still applies) unless
you override with ``--top_k``.

Examples (from repo root):

  python scripts/inference_finetuned_qwen.py --dataset data/vcot_dataset_unique.json --index 0

  python scripts/inference_finetuned_qwen.py \\
    --image EDA/eda_clicks_chart.png \\
    --prompt "What is the largest bar?"

  # Match old greedy eval behaviour
  python scripts/inference_finetuned_qwen.py --dataset data/vcot_dataset_unique.json --index 0 --greedy
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

_REPO = Path(__file__).resolve().parent.parent

_SEP = " | <sep> |"
_RE_CLICK = re.compile(r"<click>([^<]+)</click>")

_DEFAULT_SYSTEM = (
    "You are a visual chain-of-thought assistant. "
    "For every question, output ONLY a sequence of clicks on the chart regions relevant to answering the question. "
    "Use this exact format: <click>x,y</click><click>x,y</click>... "
    "Coordinates are in a normalized 0-1000 scale where (0,0) is the top-left and (1000,1000) is the bottom-right of the image. "
    "Do not output any text, explanation, or numerical answer. Only output <click> tags."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen2-VL + LoRA click inference",
        epilog=(
            "Use either (--dataset and --index) to load image+prompt from a JSON row, "
            "or both --image and --prompt explicitly."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset",
        metavar="JSON",
        default=argparse.SUPPRESS,
        help="Dataset JSON (list of dicts with image, prompt, target). Use with --index.",
    )
    src.add_argument(
        "--image",
        default=argparse.SUPPRESS,
        help="Chart image path (use with --prompt, not with --dataset).",
    )

    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index when using --dataset (0-based).",
    )
    p.add_argument(
        "--prompt",
        help="Question when using --image (ignored when using --dataset).",
    )
    p.add_argument(
        "--show_gt",
        action="store_true",
        help="When using --dataset, print ground-truth click prefix from target (before | <sep> |).",
    )
    p.add_argument(
        "--adapter",
        default="runs/qwen_lora_v3/best",
        help="Directory with adapter_model.safetensors and tokenizer (LoRA checkpoint).",
    )
    p.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max generated tokens. Click-only outputs are usually <256; lower reduces junk repetition tail.",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy decode (temp=0, no top_p sampling, repetition_penalty=1). Same as train_updated generation_eval defaults; often repeats one click.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature. Ignored when --greedy. Default 0.3 reduces mode collapse vs greedy.",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling. Used only when not --greedy and temperature>0.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k filter; 0 disables (recommended when sampling — hub default is 1). "
        "Ignored when --greedy.",
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help=">1 penalises repeating tokens (helps repeated <click>500,500</click>). Ignored when --greedy.",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="If set, caps image resolution (same idea as training). None = processor default.",
    )
    p.add_argument(
        "--system_prompt",
        type=str,
        default=_DEFAULT_SYSTEM,
        help="System message. Pass empty string to disable.",
    )
    args = p.parse_args()

    if args.greedy:
        args.temperature = 0.0
        args.repetition_penalty = 1.0

    if hasattr(args, "dataset"):
        if args.prompt:
            p.error("--prompt is set only for --image mode; use the JSON row when using --dataset.")
    else:
        if not args.prompt:
            p.error("--prompt is required when using --image.")
    return args


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_REPO / path).resolve()


def load_json_rows(path: str) -> list[dict]:
    p = Path(path) if os.path.isabs(path) else _REPO / path
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _image_path_from_row(row: dict) -> Path:
    img = row["image"]
    if os.path.isabs(img):
        return Path(img).resolve()
    return (_REPO / img).resolve()


def _gt_clicks_prefix(target: str) -> str:
    sep_pos = target.find(_SEP)
    if sep_pos != -1:
        return target[:sep_pos].strip()
    return target.strip()


def resolve_image_and_prompt(args: argparse.Namespace) -> tuple[Path, str, str | None]:
    """Returns (image_path, prompt, optional_gt_clicks_str for printing)."""
    if hasattr(args, "dataset"):
        rows = load_json_rows(args.dataset)
        if not rows:
            raise ValueError(f"Dataset is empty: {args.dataset}")
        if args.index < 0 or args.index >= len(rows):
            raise IndexError(
                f"--index {args.index} out of range (dataset has {len(rows)} rows)"
            )
        row = rows[args.index]
        if "image" not in row or "prompt" not in row:
            raise KeyError("Dataset row must have 'image' and 'prompt' keys")
        img_path = _image_path_from_row(row)
        prompt = row["prompt"]
        gt: str | None = None
        if args.show_gt and row.get("target"):
            gt = _gt_clicks_prefix(row["target"])
        return img_path, prompt, gt

    img_path = _resolve_path(args.image)
    return img_path, args.prompt, None


def _maybe_set_max_pixels(processor: Qwen2VLProcessor, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels


def run_generate(
    model: PeftModel,
    processor: Qwen2VLProcessor,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    msgs: list[dict] = []
    if args.system_prompt:
        msgs.append({"role": "system", "content": args.system_prompt})
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    })

    prompt_text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text], images=[image], return_tensors="pt"
    ).to(device)

    gen_kw: dict = {"max_new_tokens": args.max_new_tokens}
    if args.temperature and args.temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = args.temperature
        gen_kw["top_p"] = args.top_p
        # Hub defaults top_k=1 → only one candidate per step; sampling behaves like greedy.
        gen_kw["top_k"] = args.top_k
    else:
        gen_kw["do_sample"] = False
    if args.repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = args.repetition_penalty

    out_ids = model.generate(**inputs, **gen_kw)
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True)


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    img_path, prompt, gt_clicks = resolve_image_and_prompt(args)

    if not img_path.is_file():
        hint = ""
        if hasattr(args, "image") and (
            "path/to" in args.image or args.image.endswith("chart.png")
        ):
            hint = (
                " (Did you copy the docs example literally? Use a real file or --dataset ... --index N.)"
            )
        raise FileNotFoundError(f"Image not found: {img_path}{hint}")

    adapter_dir = _resolve_path(args.adapter)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"No LoRA adapter at {adapter_dir}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    dtype = (
        torch.float16 if device.type == "mps"
        else torch.bfloat16 if device.type == "cuda"
        else torch.float32
    )
    print(f"Device: {device}  dtype: {dtype}")

    processor = Qwen2VLProcessor.from_pretrained(str(adapter_dir), use_fast=False)
    _maybe_set_max_pixels(processor, args.max_pixels)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=dtype
    ).to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    image = Image.open(img_path).convert("RGB")

    if args.temperature and args.temperature > 0:
        gen_desc = (
            f"sample  temp={args.temperature}  top_p={args.top_p}  top_k={args.top_k}  "
            f"rep_penalty={args.repetition_penalty}"
        )
    else:
        gen_desc = f"greedy  rep_penalty={args.repetition_penalty}"
    print(f"Generate: {gen_desc}  max_new_tokens={args.max_new_tokens}")

    generated = run_generate(model, processor, device, image, prompt, args)

    clicks = _RE_CLICK.findall(generated)
    uniq = len({c.strip() for c in clicks})
    if clicks:
        print(f"Parsed: {len(clicks)} <click> tags, {uniq} distinct coordinate strings.")
        if uniq == 1 and len(clicks) >= 5:
            print(
                "  Note: single coordinate repeated — avoid --greedy; try default temp/top_p/rep_penalty "
                "or tune --temperature / --repetition_penalty."
            )
    else:
        print("Parsed: no <click>...</click> tags in output.")

    if hasattr(args, "dataset"):
        print(f"Dataset: {args.dataset}  index: {args.index}")
    print(f"Image: {img_path} ({image.width}x{image.height})")
    print(f"Prompt: {prompt}")
    if gt_clicks is not None:
        print(f"GT (clicks): {gt_clicks}")
    print("---")
    print(generated)


if __name__ == "__main__":
    main()
