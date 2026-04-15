#!/usr/bin/env python3
"""QA with Qwen2-VL: image + question + click path (predicted or ground-truth).

Use ``--click_field gt_clicks`` on the same eval JSON (e.g. ``Point_head_eval_full.json``)
to condition on **human ground-truth** clicks (pixels → 0–1000), for comparison with
``pred_clicks`` runs."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

_REPO = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM = (
    "You are a helpful assistant for chart question answering. "
    "You receive an image, a question, and a suggested viewing path as "
    "<click>x,y</click> tags. Coordinates use a normalized 0–1000 scale: "
    "(0,0) is the top-left and (1000,1000) is the bottom-right of the image. "
    "Treat the path as a hint about where to look. Reply with a direct, concise answer only. "
    "Do not repeat the click sequence unless asked."
)

DEFAULT_USER_SUFFIX = (
    "\n\nA model suggested the following viewing path on the chart "
    "(normalized coordinates; each tag is one location):\n{click_tokens}\n\n"
    "Answer the question based on the chart image and this path."
)

DEFAULT_GT_USER_SUFFIX = (
    "\n\nThe following is the **ground-truth** human viewing path on this chart "
    "(normalized 0–1000 coordinates; each tag is one fixation):\n{click_tokens}\n\n"
    "Answer the question based on the chart image and this path."
)

_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")
_SEP = " | <sep> | "


def parse_clicks_norm_from_target(target: str) -> list[tuple[float, float]]:
    """Clicks from dataset ``target`` string (0–1000), before `` | <sep> | ``."""
    t = target.strip()
    if _SEP in t:
        t = t.split(_SEP, 1)[0].strip()
    return [(float(x), float(y)) for x, y in _RE_CLICK.findall(t)]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_REPO / path).resolve()


def _maybe_set_max_pixels(processor: Qwen2VLProcessor, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels


def pixels_to_norm_1000(
    pred_clicks: list,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    out: list[tuple[float, float]] = []
    for pair in pred_clicks:
        x_px, y_px = float(pair[0]), float(pair[1])
        x_n = x_px / width * 1000.0
        y_n = y_px / height * 1000.0
        out.append((x_n, y_n))
    return out


def format_click_tokens(norm_clicks: list[tuple[float, float]]) -> str:
    parts = []
    for x, y in norm_clicks:
        parts.append(f"<click>{int(round(x))},{int(round(y))}</click>")
    return " ".join(parts) if parts else "(no clicks provided)"


@torch.no_grad()
def run_generate_qa(
    model: torch.nn.Module,
    processor: Qwen2VLProcessor,
    device: torch.device,
    image: Image.Image,
    user_text: str,
    *,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
) -> str:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    })
    prompt_text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text], images=[image], return_tensors="pt"
    ).to(device)

    gen_kw: dict = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = 0.95
    else:
        gen_kw["do_sample"] = False
    if repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = repetition_penalty

    out_ids = model.generate(**inputs, **gen_kw)
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QA inference with image + question + pred_clicks from eval JSON.",
    )
    p.add_argument(
        "--eval_json",
        required=True,
        help="Eval JSON with summary + per_sample (e.g. Point_head_eval_full.json).",
    )
    p.add_argument(
        "--out_json",
        required=True,
        help="Output path for results (summary + per_sample with generated answers).",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA dir (adapter_config.json). Omit to use base model only.",
    )
    p.add_argument(
        "--base_model",
        default=None,
        help="HF base model id. Default: from adapter's train_config.json, else Qwen/Qwen2-VL-2B-Instruct.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max_pixels", type=int, default=401408)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument(
        "--system_prompt",
        default=None,
        help="Override default QA system prompt.",
    )
    p.add_argument(
        "--user_suffix",
        default=None,
        help="Template with {click_tokens} placeholder appended after the question.",
    )
    p.add_argument(
        "--click_field",
        choices=("pred_clicks", "gt_clicks"),
        default="pred_clicks",
        help="Which click list to use from each eval sample (default: pred_clicks).",
    )
    p.add_argument(
        "--test_json",
        default=None,
        help="Optional: test rows with ``target`` (e.g. test_holdout.json). "
        "If set, ground-truth clicks are parsed from ``target`` (0–1000) matched by "
        "(image, prompt), and --click_field is ignored for coordinates.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = _resolve(args.eval_json)
    if not eval_path.is_file():
        print(f"ERROR: eval JSON not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    with open(eval_path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "per_sample" in payload:
        samples: list[dict] = payload["per_sample"]
        base_summary = payload.get("summary", {})
    elif isinstance(payload, list):
        samples = payload
        base_summary = {}
    else:
        print("ERROR: expected dict with per_sample or a list of samples", file=sys.stderr)
        sys.exit(1)

    samples = samples[args.offset :]
    if args.limit is not None:
        samples = samples[: args.limit]

    adapter_dir: Path | None = _resolve(args.adapter) if args.adapter else None
    if adapter_dir is not None and not (adapter_dir / "adapter_config.json").exists():
        print(f"ERROR: no LoRA adapter at {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    train_cfg: dict = {}
    if adapter_dir is not None:
        tc = adapter_dir / "train_config.json"
        if tc.is_file():
            with open(tc, encoding="utf-8") as f:
                train_cfg = json.load(f)

    base_model = (
        args.base_model
        or train_cfg.get("model_name")
        or "Qwen/Qwen2-VL-2B-Instruct"
    )
    max_pixels = args.max_pixels if args.max_pixels is not None else train_cfg.get("max_pixels")

    system_prompt = args.system_prompt if args.system_prompt is not None else DEFAULT_SYSTEM
    if args.user_suffix is not None:
        user_suffix = args.user_suffix
    elif args.test_json is not None:
        user_suffix = DEFAULT_GT_USER_SUFFIX
    elif args.click_field == "gt_clicks":
        user_suffix = DEFAULT_GT_USER_SUFFIX
    else:
        user_suffix = DEFAULT_USER_SUFFIX

    test_by_key: dict[tuple[str, str], str] | None = None
    if args.test_json is not None:
        test_path = _resolve(args.test_json)
        if not test_path.is_file():
            print(f"ERROR: --test_json not found: {test_path}", file=sys.stderr)
            sys.exit(1)
        with open(test_path, encoding="utf-8") as f:
            test_rows = json.load(f)
        test_by_key = {}
        for r in test_rows:
            img = (r.get("image") or "").strip()
            pr = (r.get("prompt") or "").strip()
            tgt = r.get("target") or ""
            if img and pr:
                test_by_key[(img, pr)] = tgt
        print(f"Loaded {len(test_by_key)} (image,prompt) targets from {test_path}")

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
    print(f"Eval:   {eval_path}  (n={len(samples)})")
    print(
        f"base_model={base_model}  adapter={adapter_dir}  max_pixels={max_pixels}  "
        f"click_field={args.click_field}"
        + (f"  test_json={args.test_json}" if args.test_json else "")
    )

    if adapter_dir is not None:
        processor = Qwen2VLProcessor.from_pretrained(str(adapter_dir), use_fast=False)
        _maybe_set_max_pixels(processor, int(max_pixels) if max_pixels is not None else None)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype
        ).to(device)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    else:
        processor = Qwen2VLProcessor.from_pretrained(base_model, use_fast=False)
        _maybe_set_max_pixels(processor, int(max_pixels) if max_pixels is not None else None)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype
        ).to(device)

    model.eval()

    results: list[dict] = []
    for s in tqdm(samples, desc="qa"):
        idx = s.get("index", len(results))
        img_rel = s["image"]
        prompt = s["prompt"]

        img_path = Path(img_rel) if os.path.isabs(img_rel) else _REPO / img_rel
        image = Image.open(img_path).convert("RGB")
        w, h = image.width, image.height

        if test_by_key is not None:
            tgt = test_by_key.get(((str(img_rel)).strip(), (prompt or "").strip()))
            if tgt is None:
                print(f"ERROR: no test row for image+prompt index={idx}", file=sys.stderr)
                sys.exit(1)
            norm_clicks = parse_clicks_norm_from_target(tgt)
            clicks_px: list = [
                [round(x / 1000.0 * w, 2), round(y / 1000.0 * h, 2)] for x, y in norm_clicks
            ]
            click_source = "target"
        else:
            clicks_px = s.get(args.click_field) or []
            norm_clicks = pixels_to_norm_1000(clicks_px, w, h)
            click_source = args.click_field

        click_tokens = format_click_tokens(norm_clicks)
        user_text = prompt + user_suffix.format(click_tokens=click_tokens)

        answer = run_generate_qa(
            model,
            processor,
            device,
            image,
            user_text,
            system_prompt=system_prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            repetition_penalty=float(args.repetition_penalty),
        )

        results.append({
            "index": idx,
            "image": str(img_rel),
            "prompt": prompt,
            "click_source": click_source,
            "clicks_pixel": clicks_px,
            "clicks_norm_1000": [[round(x, 4), round(y, 4)] for x, y in norm_clicks],
            "answer": answer.strip(),
        })

    out_summary = {
        **base_summary,
        "eval_json": str(eval_path),
        "out_json": str(_resolve(args.out_json)),
        "n_samples": len(results),
        "base_model": base_model,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "max_pixels": max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "click_field": args.click_field,
        "test_json": str(_resolve(args.test_json)) if args.test_json else None,
    }

    out_path = _resolve(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": out_summary, "per_sample": results}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path} ({len(results)} samples)")


if __name__ == "__main__":
    main()
