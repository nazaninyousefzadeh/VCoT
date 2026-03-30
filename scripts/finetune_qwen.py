"""
LoRA fine-tune Qwen2-VL on VCoT JSON: {image, prompt, target}.

Cross-entropy on assistant tokens only (user prompt masked with -100).

Usage (from repo root):
  venv/bin/python scripts/finetune_qwen.py --dataset data/vcot_dataset_unique.json --output_dir runs/qwen_lora
  venv/bin/python scripts/finetune_qwen.py --limit 500 --epochs 1
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

_REPO = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/vcot_dataset.json")
    p.add_argument("--output_dir", default="runs/qwen_lora")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    return p.parse_args()


class VCoTJsonDataset(Dataset):
    def __init__(self, path: str, limit: int | None):
        with open(_REPO / path if not os.path.isabs(path) else path) as f:
            self.rows = json.load(f)
        if limit is not None:
            self.rows = self.rows[:limit]

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
        # Right-padded batch: find real start of sequence
        row = labels[i]
        ids = input_ids[i]
        non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
        if len(non_pad) == 0:
            continue
        start = int(non_pad[0])
        pr = pt["input_ids"][0]
        # Match prompt prefix from first real token
        j = 0
        while j < plen and start + j < len(ids) and ids[start + j] == pr[j]:
            j += 1
        labels[i, : start + j] = -100

    if pad_id is not None:
        labels[labels == pad_id] = -100

    batch["labels"] = labels
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main():
    args = parse_args()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    ds = VCoTJsonDataset(args.dataset, args.limit)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for raw in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            batch = build_batch(raw, processor, device)
            opt.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"epoch {epoch+1} mean loss: {total / max(n, 1):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
