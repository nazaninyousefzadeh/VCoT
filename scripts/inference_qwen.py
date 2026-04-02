"""
Run Qwen2-VL inference on the VCoT dataset.

Passes each question + chart image to the model and saves responses (one sample
at a time).

Dataset `target` format: "{clicks} | <sep> | {answer}" — the final answer string
is after `` | <sep> | `` (see preprocessing/vcot_target.py).

Usage:
  venv/bin/python scripts/inference_qwen.py --output data/qwen_responses.json
  venv/bin/python scripts/inference_qwen.py --output data/qwen_responses.json --limit 1000
  venv/bin/python scripts/inference_qwen.py --output data/qwen_responses.json --start 500 --limit 200
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

_PREPROCESSING = Path(__file__).resolve().parent.parent / "preprocessing"
if str(_PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING))
from vcot_target import parse_vcot_target


def _ground_truth_fields(target: str) -> dict:
    """Full target plus parsed click sequence and answer (after `` | <sep> | ``)."""
    clicks, answer = parse_vcot_target(target)
    return {
        "ground_truth": target,
        "ground_truth_clicks": clicks,
        "ground_truth_answer": answer,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/vcot_dataset_unique.json")
    p.add_argument("--output", default="data/qwen_responses.json")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    p.add_argument("--limit", type=int, default=None, help="Max samples to process")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    end = args.start + args.limit if args.limit else len(data)
    data = data[args.start:end]
    print(f"Processing samples {args.start} to {args.start + len(data)}")

    # Load existing results if resuming
    results = []
    if os.path.exists(args.output) and args.start > 0:
        with open(args.output) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    print(f"Loading model: {args.model_name}")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    print(f"Model loaded on {device}")

    for i, sample in enumerate(tqdm[Any](data, desc="Inference")):
        idx = args.start + i
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except Exception as e:
            results.append(
                {
                    "index": idx,
                    "image": sample["image"],
                    "prompt": f"Answer the question about this chart in as few words as possible.\n\n{sample['prompt']}",
                    **_ground_truth_fields(sample["target"]),
                    "response": None,
                    "error": str(e),
                }
            )
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Answer the question about this chart in as few words as possible.\n\n{sample['prompt']}"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        results.append(
            {
                "index": idx,
                "image": sample["image"],
                "prompt": f"Answer the question about this chart in as few words as possible.\n\n{sample['prompt']}",
                **_ground_truth_fields(sample["target"]),
                "response": response,
            }
        )

        if (i + 1) % 100 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f)
            print(f"\nSaved {len(results)} results so far")

    with open(args.output, "w") as f:
        json.dump(results, f)

    print(f"\nDone! Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
