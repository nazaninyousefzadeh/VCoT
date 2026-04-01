"""
LoRA fine-tune Qwen2-VL on VCoT JSON: {image, prompt, target}.

Cross-entropy on assistant tokens only (user prompt masked with -100).

Default: shuffle split into train / val / test; track val loss; save the lowest-val
checkpoint to ``output_dir/best`` and the final epoch to ``output_dir/last``.
Held-out test rows are saved as ``output_dir/test_holdout.json``.

Usage (from repo root):
  venv/bin/python scripts/finetune_qwen.py --dataset data/vcot_dataset_unique.json --output_dir runs/qwen_lora
  venv/bin/python scripts/finetune_qwen.py --epochs 3 --batch_size 1 --gradient_accumulation_steps 8

Memory tips (laptop / MPS): use ``--batch_size 1``, increase ``--gradient_accumulation_steps``,
``--gradient_checkpointing`` (on by default), optional ``--max_pixels``, and smaller ``--lora_r``.

Inference with best adapter:
  venv/bin/python scripts/inference_qwen.py --adapter_path runs/qwen_lora/best ...
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

_REPO = Path(__file__).resolve().parent.parent


def _maybe_set_max_pixels(processor: AutoProcessor, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
        print(f"Set image_processor.max_pixels={max_pixels}")
    else:
        print("Warning: could not set max_pixels on this processor; ignoring --max_pixels")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/vcot_dataset.json")
    p.add_argument("--output_dir", default="runs/qwen_lora")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Microbatch size; lower if OOM (try 1 on laptop).",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Optimizer step every N microbatches; effective batch ≈ batch_size * N.",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--limit", type=int, default=None, help="Use only first N rows (debug)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute activations during backward (much lower memory; slightly slower).",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="If set, caps Qwen image_processor.max_pixels (e.g. 501760 ≈ 640*28*28; lower = less VRAM).",
    )
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_json_rows(path: str) -> list[dict]:
    with open(_REPO / path if not os.path.isabs(path) else path) as f:
        return json.load(f)


def train_val_test_split(
    rows: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1")
    n = len(rows)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Split leaves no training rows; reduce val/test ratios or use more data")
    train_i = idx[n_test + n_val :]
    val_i = idx[n_test : n_test + n_val]
    test_i = idx[:n_test]
    return [rows[i] for i in train_i], [rows[i] for i in val_i], [rows[i] for i in test_i]


class VCoTJsonDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(_REPO / r["image"] if not os.path.isabs(r["image"]) else r["image"]).convert(
            "RGB"
        )
        return {"image": img, "prompt": r["prompt"], "target": r["target"]}


def build_batch(samples: list[dict], processor: AutoProcessor, device: torch.device) -> dict:
    """Tokenize user+assistant; mask labels on user/prefix tokens."""
    images = [s["image"] for s in samples]
    prompt_texts = []
    full_texts = []
    for s in samples:
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text", "text": s["prompt"]},
            ],
        }
        full_msgs = [
            user,
            {"role": "assistant", "content": [{"type": "text", "text": s["target"]}]},
        ]
        prompt_texts.append(
            processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        )
        full_texts.append(
            processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        )

    batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]
    labels = input_ids.clone()

    pad_id = processor.tokenizer.pad_token_id
    for i in range(len(samples)):
        pt = processor(
            text=[prompt_texts[i]], images=[images[i]], return_tensors="pt", padding=False
        )
        plen = pt["input_ids"].shape[1]
        row = labels[i]
        ids = input_ids[i]
        non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
        if len(non_pad) == 0:
            continue
        start = int(non_pad[0])
        pr = pt["input_ids"][0]
        j = 0
        while j < plen and start + j < len(ids) and ids[start + j] == pr[j]:
            j += 1
        labels[i, : start + j] = -100

    if pad_id is not None:
        labels[labels == pad_id] = -100

    batch["labels"] = labels
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def mean_loss_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    processor: AutoProcessor,
    device: torch.device,
) -> float | None:
    if len(loader) == 0:
        return None
    model.eval()
    total, batches = 0.0, 0
    for raw in loader:
        batch = build_batch(raw, processor, device)
        out = model(**batch)
        total += float(out.loss.item())
        batches += 1
    model.train()
    return total / max(batches, 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    rows = load_json_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]

    train_rows, val_rows, test_rows = train_val_test_split(
        rows, args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"Split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"(seed={args.seed})"
    )

    train_loader = DataLoader(
        VCoTJsonDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
    )
    val_loader = DataLoader(
        VCoTJsonDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: b,
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
    _maybe_set_max_pixels(processor, args.max_pixels)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(device)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (use_cache=False)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "split_manifest.json", "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "train_n": len(train_rows),
                "val_n": len(val_rows),
                "test_n": len(test_rows),
            },
            f,
            indent=2,
        )
    test_path = out_root / "test_holdout.json"
    with open(test_path, "w") as f:
        json.dump(test_rows, f)
    print(f"Wrote held-out test: {test_path} ({len(test_rows)} samples)")

    best_dir = out_root / "best"
    last_dir = out_root / "last"
    best_val: float | None = None

    model.train()
    accum = max(1, args.gradient_accumulation_steps)
    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, raw in enumerate(pbar):
            batch = build_batch(raw, processor, device)
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            total += out.loss.item()
            n += 1
            stepped = (step + 1) % accum == 0 or step == len(train_loader) - 1
            if stepped:
                opt.step()
                opt.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")

        train_mean = total / max(n, 1)
        val_mean = mean_loss_on_loader(model, val_loader, processor, device)
        if val_mean is None:
            print(f"epoch {epoch+1} train loss: {train_mean:.4f}  val loss: (no val set)")
        else:
            print(f"epoch {epoch+1} train loss: {train_mean:.4f}  val loss: {val_mean:.4f}")

        if val_mean is not None and not math.isnan(val_mean):
            if best_val is None or val_mean < best_val:
                best_val = val_mean
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                with open(best_dir / "best_metrics.json", "w") as f:
                    json.dump(
                        {"epoch": epoch + 1, "val_loss": val_mean, "train_loss": train_mean},
                        f,
                        indent=2,
                    )
                print(f"  saved new best -> {best_dir} (val_loss={val_mean:.4f})")

    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    print(f"Saved last epoch -> {last_dir}")
    if best_val is not None:
        print(f"Best val_loss={best_val:.4f} in {best_dir}")
    else:
        print("No validation loss tracked; use last/ or expand val set.")


if __name__ == "__main__":
    main()
