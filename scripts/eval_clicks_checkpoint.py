#!/usr/bin/env python3
"""Evaluate a saved LoRA checkpoint on a test JSON (e.g. runs/.../test_holdout.json).

Generates click sequences with the fine-tuned LM and compares to ground-truth clicks
(DTW, length error, optional mean L2 on matched prefix). The auxiliary PointHead is
training-only; inference uses ``model.generate`` like training-time gen eval.

Example (after epoch 1 finishes):

  python scripts/eval_clicks_checkpoint.py \\
    --adapter runs/point_head_2ep/epoch_0001 \\
    --test_json runs/point_head_2ep/test_holdout.json \\
    --max_pixels 200704 \\
    --out_dataset_json runs/point_head_2ep/pred_clicks_dataset.json

Use ``--out_dataset_json`` to save predicted ``<click>x,y</click>`` targets for
``scripts/plot_click_arrows.py``.

If ``train_config.json`` exists under ``--adapter``, defaults for system prompt and
``max_pixels`` are taken from there unless overridden.
"""
from __future__ import annotations

import argparse
import json
import math
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
_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")
_SEP = " | <sep> |"


def dtw_distance(
    seq_a: list[tuple[float, float]],
    seq_b: list[tuple[float, float]],
) -> float:
    if not seq_a and not seq_b:
        return 0.0
    if not seq_a or not seq_b:
        return float("inf")
    n, m = len(seq_a), len(seq_b)
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    def dist(p, q):
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dist(seq_a[i - 1], seq_b[j - 1]) + min(
                dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]
            )
    return dp[n][m] / (n + m)


def parse_clicks(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in _RE_CLICK.findall(text)]


def gt_clicks_from_target(target: str) -> list[tuple[float, float]]:
    t = target
    sep_pos = t.find(_SEP)
    if sep_pos != -1:
        t = t[:sep_pos].strip()
    return parse_clicks(t)


def format_clicks_as_target(
    clicks: list[tuple[float, float]],
    answer: str | None = None,
) -> str:
    """Build a dataset-style ``target`` string: ``<click>x,y</click>`` tokens, optional ``| <sep> |`` answer."""
    parts = [f"<click>{int(round(x))},{int(round(y))}</click>" for x, y in clicks]
    s = " ".join(parts)
    if answer is not None and str(answer).strip() != "":
        s = s + _SEP + str(answer).strip()
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        required=True,
        help="Checkpoint dir with adapter + tokenizer (e.g. runs/point_head_2ep/epoch_0001).",
    )
    p.add_argument(
        "--test_json",
        default=None,
        help="Test rows (image, prompt, target). Default: <run_dir>/test_holdout.json next to adapter.",
    )
    p.add_argument("--base_model", default=None, help="Override base HF model id.")
    p.add_argument("--limit", type=int, default=None, help="Max number of test rows.")
    p.add_argument("--offset", type=int, default=0, help="Skip first N rows.")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument(
        "--system_prompt",
        default=None,
        help="Override system prompt (default: from train_config.json or train_point_head default).",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Write per-sample metrics to this JSON file.",
    )
    p.add_argument(
        "--out_dataset_json",
        default=None,
        help="Write predictions as a dataset-style JSON list [{image, prompt, target}, ...] "
        "where target is predicted <click>x,y</click>... (for scripts/plot_click_arrows.py).",
    )
    return p.parse_args()


def _resolve(p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_REPO / path).resolve()


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
            {"type": "text", "text": prompt},
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
        gen_kw["top_k"] = 0
    else:
        gen_kw["do_sample"] = False
    if repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = repetition_penalty

    out_ids = model.generate(**inputs, **gen_kw)
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    adapter_dir = _resolve(args.adapter)
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"ERROR: no LoRA adapter at {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    run_dir = adapter_dir.parent
    test_path = _resolve(args.test_json) if args.test_json else (run_dir / "test_holdout.json")
    if not test_path.is_file():
        print(f"ERROR: test JSON not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    train_cfg_path = adapter_dir / "train_config.json"
    train_cfg: dict = {}
    if train_cfg_path.is_file():
        with open(train_cfg_path, encoding="utf-8") as f:
            train_cfg = json.load(f)

    base_model = args.base_model or train_cfg.get("model_name") or "Qwen/Qwen2-VL-2B-Instruct"
    max_pixels = args.max_pixels if args.max_pixels is not None else train_cfg.get("max_pixels")
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else train_cfg.get("gen_max_new_tokens", 512)
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else train_cfg.get("gen_temperature", 0.3)
    )
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else train_cfg.get("gen_repetition_penalty", 1.0)
    )
    if args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = train_cfg.get(
            "system_prompt",
            (
                "You are a visual chain-of-thought assistant. "
                "For every question, output ONLY a sequence of clicks on the chart regions relevant to answering the question. "
                "Use this exact format: <click>x,y</click><click>x,y</click>... "
                "Coordinates are in a normalized 0-1000 scale where (0,0) is the top-left and (1000,1000) is the bottom-right of the image. "
                "Do not output any text, explanation, or numerical answer. Only output <click> tags."
            ),
        )

    with open(test_path, encoding="utf-8") as f:
        rows: list[dict] = json.load(f)

    slice_rows = rows[args.offset :]
    if args.limit is not None:
        slice_rows = slice_rows[: args.limit]

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
    print(f"Adapter: {adapter_dir}")
    print(f"Test:    {test_path}  (n={len(slice_rows)})")
    print(f"base_model={base_model}  max_pixels={max_pixels}  max_new_tokens={max_new_tokens}")
    print(f"temp={temperature}  rep_penalty={repetition_penalty}")

    processor = Qwen2VLProcessor.from_pretrained(str(adapter_dir), use_fast=False)
    _maybe_set_max_pixels(processor, int(max_pixels) if max_pixels is not None else None)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=dtype
    ).to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    results: list[dict] = []
    dataset_rows: list[dict] = []
    dtw_list: list[float] = []
    len_err_list: list[float] = []

    for i, row in enumerate(tqdm(slice_rows, desc="test")):
        img_rel = row["image"]
        img_path = Path(img_rel) if os.path.isabs(img_rel) else _REPO / img_rel
        prompt = row["prompt"]
        target = row.get("target", "")
        gt = gt_clicks_from_target(target)

        image = Image.open(img_path).convert("RGB")
        generated = run_generate(
            model,
            processor,
            device,
            image,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
        )
        pred = parse_clicks(generated)
        dtw = dtw_distance(pred, gt)
        len_err = abs(len(pred) - len(gt)) / max(len(gt), 1)
        if not math.isinf(dtw):
            dtw_list.append(dtw)
        len_err_list.append(len_err)

        mean_l2: float | None = None
        if pred and gt:
            n_m = min(len(pred), len(gt))
            mean_l2 = sum(
                math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                for (px, py), (gx, gy) in zip(pred[:n_m], gt[:n_m])
            ) / n_m

        rec = {
            "index": args.offset + i,
            "image": str(img_rel),
            "prompt": prompt,
            "n_pred": len(pred),
            "n_gt": len(gt),
            "dtw": dtw,
            "len_err": len_err,
            "mean_l2_matched": mean_l2,
            "pred_clicks": [list(p) for p in pred],
            "gt_clicks": [list(p) for p in gt],
            "raw_output_prefix": generated[:500],
        }
        results.append(rec)
        dataset_rows.append(
            {
                "image": str(img_rel),
                "prompt": prompt,
                "target": format_clicks_as_target(pred),
            }
        )

        tqdm.write(
            f"[{args.offset + i}] n_pred={len(pred)} n_gt={len(gt)}  "
            f"DTW={dtw:.2f}  len_err={len_err:.3f}"
            + (f"  mean_L2={mean_l2:.1f}" if mean_l2 is not None else "")
        )

    finite = [d for d in dtw_list if not math.isinf(d)]
    summary = {
        "adapter": str(adapter_dir),
        "test_json": str(test_path),
        "n_samples": len(results),
        "mean_dtw": sum(finite) / len(finite) if finite else None,
        "mean_len_err": sum(len_err_list) / len(len_err_list) if len_err_list else None,
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.out_json:
        out_p = _resolve(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2)
        print(f"Wrote {out_p}")

    if args.out_dataset_json:
        ds_p = _resolve(args.out_dataset_json)
        ds_p.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_p, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, indent=2, ensure_ascii=False)
        print(f"Wrote dataset JSON ({len(dataset_rows)} rows) -> {ds_p}")


if __name__ == "__main__":
    main()
