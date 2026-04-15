
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import random
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
)

_REPO = Path(__file__).resolve().parent.parent
_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")
_ASST_HEADER = "<|im_start|>assistant"
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


# ─────────────────────────────────────────────────────────────────────────────
# PointHead — the AutoTraces-style lightweight coordinate decoder
# ─────────────────────────────────────────────────────────────────────────────

class PointHead(nn.Module):


    def __init__(self, hidden_size: int, intermediate: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (N, hidden_size) → (N, 2) predicted (x, y) in raw coordinate scale"""
        return self.net(h)


def trainable_parameter_counts(
    model: nn.Module, point_head: Optional[PointHead]
) -> dict[str, int]:
    def count_trainable(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    out: dict[str, int] = {"model_trainable": count_trainable(model)}
    if point_head is not None:
        out["point_head_trainable"] = count_trainable(point_head)
        out["total_trainable"] = out["model_trainable"] + out["point_head_trainable"]
    else:
        out["total_trainable"] = out["model_trainable"]
    return out


def json_safe_for_metrics(obj: object) -> object:
    """Make structures JSON-serialisable (NaN/Inf → null)."""
    if isinstance(obj, dict):
        return {k: json_safe_for_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe_for_metrics(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _git_revision(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def save_run_checkpoint(
    out_dir: Path,
    model,
    processor: Qwen2VLProcessor,
    point_head: Optional[PointHead],
    metrics: dict,
    train_config_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    if point_head is not None:
        torch.save(point_head.state_dict(), out_dir / "point_head.pt")
    with open(out_dir / "epoch_metrics.json", "w") as f:
        json.dump(json_safe_for_metrics(metrics), f, indent=2)
    if train_config_path.is_file():
        shutil.copy2(train_config_path, out_dir / "train_config.json")


# ─────────────────────────────────────────────────────────────────────────────
# Click-position extraction
# ─────────────────────────────────────────────────────────────────────────────

def click_open_patterns(tokenizer) -> list[list[int]]:
    """Token ID runs that end at the last token of an opening `<click>` tag.

    Qwen (and similar BPE models) often tokenize the first `<click>` differently
    from ` <click>` after `</click> ` — e.g. [27,3678,29] vs [366,3678,29].
    Matching only ``encode("<click>")`` misses every click after the first.
    """
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for prefix in ("<click>", " <click>", "\n<click>", "\t<click>"):
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if not ids:
            continue
        t = tuple(ids)
        if t not in seen:
            seen.add(t)
            out.append(ids)
    out.sort(key=len, reverse=True)
    return out


def find_click_open_positions(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    open_patterns: list[list[int]],
    max_clicks: int = 0,
) -> list[int]:
    if not open_patterns:
        return []

    asst_pos = (labels != -100).nonzero(as_tuple=True)[0]
    if len(asst_pos) == 0:
        return []
    first_asst = int(asst_pos[0].item())

    min_m = min(len(p) for p in open_patterns)
    pat_tensors = [
        torch.tensor(p, device=input_ids.device, dtype=input_ids.dtype)
        for p in open_patterns
    ]
    T = len(input_ids)
    positions: list[int] = []
    i = first_asst
    while i <= T - min_m:
        matched = False
        for pt in pat_tensors:
            m = int(pt.shape[0])
            if i + m > T:
                continue
            if (input_ids[i : i + m] == pt).all():
                positions.append(i + m - 1)  # last token of opening `<click>` run
                i += m
                matched = True
                break
        if not matched:
            i += 1
        if max_clicks > 0 and len(positions) >= max_clicks:
            break
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate stats (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CoordStats(NamedTuple):
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


def compute_coord_stats(rows: list[dict], clicks_only: bool = True) -> CoordStats:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        target = r["target"]
        if clicks_only:
            sep_pos = target.find(_SEP)
            if sep_pos != -1:
                target = target[:sep_pos].strip()
        for x_str, y_str in _RE_CLICK.findall(target):
            xs.append(float(x_str))
            ys.append(float(y_str))
    if len(xs) < 2:
        return CoordStats(500.0, 500.0, 500.0, 500.0)
    xs_t = torch.tensor(xs, dtype=torch.float64)
    ys_t = torch.tensor(ys, dtype=torch.float64)
    return CoordStats(
        x_mean=float(xs_t.mean()),
        x_std=max(float(xs_t.std()), 1.0),
        y_mean=float(ys_t.mean()),
        y_std=max(float(ys_t.std()), 1.0),
    )


def compute_point_head_loss(
    last_hidden: torch.Tensor,   
    input_ids: torch.Tensor,     
    labels: torch.Tensor,         
    target_texts: list[str],
    point_head: PointHead,
    open_patterns: list[list[int]],
    coord_scale: float = 1000.0,
    max_clicks: int = 0,
    coord_stats: Optional[CoordStats] = None,
) -> Optional[torch.Tensor]:

    all_pred: list[torch.Tensor] = []
    all_gt:   list[torch.Tensor] = []
    for i, tgt in enumerate(target_texts):
        gt_clicks = parse_clicks(tgt)
        if not gt_clicks:
            continue
        if max_clicks > 0:
            gt_clicks = gt_clicks[:max_clicks]

        positions = find_click_open_positions(
            input_ids[i], labels[i], open_patterns, max_clicks=max_clicks
        )
        if not positions:
            continue

        n = min(len(positions), len(gt_clicks))
        h = last_hidden[i, positions[:n], :]     
        pred_xy = point_head(h.float())          

        gt_t = torch.tensor(
            gt_clicks[:n], dtype=pred_xy.dtype, device=pred_xy.device
        )

        
        all_pred.append(pred_xy)
        all_gt.append(gt_t)

    if not all_pred:
        return None

    pred_cat = torch.cat(all_pred, dim=0)  
    gt_cat   = torch.cat(all_gt,   dim=0)   

    if coord_stats is not None:
        scale = torch.tensor(
            [coord_stats.x_std, coord_stats.y_std],
            device=pred_cat.device, dtype=pred_cat.dtype,
        )
    else:
        scale = pred_cat.new_full((2,), coord_scale)

    return F.l1_loss(pred_cat / scale, gt_cat / scale)

class BubbleViewDataset(Dataset):
    def __init__(self, rows: list[dict], clicks_only: bool = True) -> None:
        self.rows = rows
        self.clicks_only = clicks_only

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        img_path = r["image"] if os.path.isabs(r["image"]) else str(_REPO / r["image"])
        img = Image.open(img_path).convert("RGB")
        target = r["target"]
        if self.clicks_only:
            sep_pos = target.find(_SEP)
            if sep_pos != -1:
                target = target[:sep_pos].strip()
        return {
            "image": img,
            "prompt": r["prompt"],
            "target": target,
            "img_w": img.width,
            "img_h": img.height,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Batch building (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _find_assistant_start(input_ids: torch.Tensor, tokenizer) -> int:
    header_ids = tokenizer.encode(_ASST_HEADER, add_special_tokens=False)
    n, m = len(input_ids), len(header_ids)
    pat = torch.tensor(header_ids, dtype=input_ids.dtype, device=input_ids.device)
    for i in range(n - m, -1, -1):
        if (input_ids[i : i + m] == pat).all():
            return i + m
    warnings.warn(
        "Could not locate assistant header in token sequence. "
        "No prompt masking applied for this sample.",
        stacklevel=3,
    )
    return 0


def build_batch(
    samples: list[dict],
    processor: Qwen2VLProcessor,
    device: torch.device,
    system_prompt: str = "",
) -> dict:
    images = [s["image"] for s in samples]
    full_texts: list[str] = []
    for s in samples:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs += [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": s["image"]},
                    {"type": "text",  "text":  s["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": s["target"]}],
            },
        ]
        full_texts.append(
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        )
    batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]
    labels = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id
    for i in range(len(samples)):
        asst_start = _find_assistant_start(input_ids[i], processor.tokenizer)
        labels[i, :asst_start] = -100
    if pad_id is not None:
        labels[labels == pad_id] = -100
    batch["labels"] = labels
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

class ClickRepetitionBlocker(LogitsProcessor):
    """Penalise digit tokens that would reproduce coordinates from recent clicks.

    Blocks two patterns that cause mode collapse:
      1. Direct repeat:    click[n] == click[n-1]   (single-coord loop)
      2. Alternating pair: click[n] == click[n-2]   (ping-pong loop)

    Both are applied independently to x and y while generating each coordinate.
    """

    _RE = re.compile(r"<click>(\d+),(\d+)</click>")

    def __init__(self, prompt_length: int, tokenizer, penalty: float = 50.0) -> None:
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.penalty = penalty

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            gen_ids = input_ids[b, self.prompt_length :]
            gen_text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
            clicks = self._RE.findall(gen_text)
            if not clicks:
                continue
            open_pos  = gen_text.rfind("<click>")
            close_pos = gen_text.rfind("</click>")
            if open_pos == -1 or open_pos <= close_pos:
                continue

            partial    = gen_text[open_pos + len("<click>"):]
            coord_idx  = 0 if "," not in partial else 1  # 0=x, 1=y

            # Block last click AND the one two steps back to break ping-pong loops
            candidates: set[str] = set()
            if len(clicks) >= 1:
                candidates.add(clicks[-1][coord_idx])
            if len(clicks) >= 2:
                candidates.add(clicks[-2][coord_idx])

            for blocked_str in candidates:
                for tid in set(self.tokenizer.encode(blocked_str, add_special_tokens=False)):
                    if 0 <= tid < scores.shape[-1]:
                        scores[b, tid] -= self.penalty
        return scores


def _gen_eval_slice(rows: list[dict], n: int, start: int) -> list[dict]:
    if not rows or n <= 0:
        return []
    L = len(rows)
    return [rows[(start + i) % L] for i in range(n)]


@torch.no_grad()
def generation_eval(
    model,
    samples: list[dict],
    processor: Qwen2VLProcessor,
    device: torch.device,
    max_new_tokens: int = 512,
    verbose: bool = True,
    system_prompt: str = "",
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    block_repeats: bool = False,
    block_repeat_penalty: float = 50.0,
) -> dict:
    model.eval()
    dtw_scores: list[float] = []
    len_errors: list[float] = []
    per_sample: list[dict]  = []

    for i, s in enumerate(samples):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text",  "text":  s["prompt"]},
            ],
        })
        prompt_text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt_text], images=[s["image"]], return_tensors="pt"
        ).to(device)

        gen_kw: dict = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
            gen_kw["top_k"] = 0
        else:
            gen_kw["do_sample"] = False
        if repetition_penalty > 1.0:
            gen_kw["repetition_penalty"] = repetition_penalty
        if block_repeats:
            prompt_len = inputs["input_ids"].shape[1]
            blocker = ClickRepetitionBlocker(
                prompt_len, processor.tokenizer, penalty=block_repeat_penalty
            )
            gen_kw["logits_processor"] = LogitsProcessorList([blocker])

        out_ids  = model.generate(**inputs, **gen_kw)
        gen_ids  = out_ids[0, inputs["input_ids"].shape[1]:]
        generated = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_clicks = parse_clicks(generated)
        gt_clicks   = parse_clicks(s["target"])
        score    = dtw_distance(pred_clicks, gt_clicks)
        len_err  = abs(len(pred_clicks) - len(gt_clicks)) / max(len(gt_clicks), 1)
        dtw_scores.append(score)
        len_errors.append(len_err)

        img_w = s.get("img_w", 0)
        img_h = s.get("img_h", 0)
        img_label = s["image"] if isinstance(s["image"], str) else f"PIL({img_w}x{img_h})"
        per_sample.append({
            "idx": i, "image": img_label, "prompt": s["prompt"],
            "gt_clicks": gt_clicks, "pred_clicks": pred_clicks,
            "dtw": score, "len_err": len_err,
        })

        if verbose:
            prompt_toks = inputs["input_ids"].shape[1]
            gen_toks    = len(gen_ids)
            # Rough image-token estimate: total prompt minus text-only re-encode
            text_only_toks = len(processor.tokenizer.encode(
                s["prompt"], add_special_tokens=False
            ))
            img_toks_approx = prompt_toks - text_only_toks
            print(f"\n[gen_eval {i+1}/{len(samples)}] {img_label}  ({img_w}x{img_h})")
            print(f"  Budget  : prompt={prompt_toks} tok  "
                  f"(~{img_toks_approx} img + ~{text_only_toks} text)  "
                  f"gen={gen_toks} tok  "
                  f"coord_tok≈{gen_toks} vs ideal={len(gt_clicks)}")
            print(f"  Prompt  : {s['prompt']}")
            print(f"  Raw out : {generated[:200]!r}")
            print(f"  GT  ({len(gt_clicks):2d} clicks): {gt_clicks}")
            print(f"  Pred({len(pred_clicks):2d} clicks): {pred_clicks}")
            # Diagnostic: separate position error from count error
            if pred_clicks and gt_clicks:
                n_matched = min(len(pred_clicks), len(gt_clicks))
                mean_l2 = sum(
                    math.sqrt((px - gx)**2 + (py - gy)**2)
                    for (px, py), (gx, gy) in zip(pred_clicks[:n_matched], gt_clicks[:n_matched])
                ) / n_matched
                print(f"  DTW={score:.2f}  len_err={len_err:.3f}  "
                      f"mean_L2(matched)={mean_l2:.1f}  "
                      f"[{'COUNT OK' if len_err < 0.15 else 'COUNT WRONG'} | "
                      f"{'POS OK' if mean_l2 < 100 else 'POS WRONG'}]")
            else:
                print(f"  DTW={score:.2f}  len_err={len_err:.3f}")

    model.train()
    return {
        "gen_dtw":     sum(dtw_scores) / len(dtw_scores) if dtw_scores else float("nan"),
        "gen_len_err": sum(len_errors) / len(len_errors) if len_errors else float("nan"),
        "per_sample":  per_sample,
    }

@torch.no_grad()
def evaluate_loader(
    model,
    loader: DataLoader,
    processor: Qwen2VLProcessor,
    device: torch.device,
    point_head: Optional[PointHead],
    open_patterns: list[list[int]],
    coord_lambda: float,
    coord_scale: float,
    max_coord_clicks: int = 0,
    system_prompt: str = "",
    max_seq_len: Optional[int] = None,
    coord_stats: Optional[CoordStats] = None,
    max_batches: Optional[int] = None,
) -> dict:
    if len(loader) == 0:
        return {"ce_loss": None, "coord_loss": None}

    model.eval()
    ce_total = coord_total = 0.0
    n = n_coord = skipped = 0
    use_coord = coord_lambda > 0 and point_head is not None

    for step, raw in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = build_batch(raw, processor, device, system_prompt=system_prompt)
        if max_seq_len is not None and batch["input_ids"].shape[1] > max_seq_len:
            skipped += 1
            del batch
            continue

        out = model(**batch, output_hidden_states=use_coord)
        ce_total += float(out.loss.item())
        n += 1

        if use_coord and out.hidden_states is not None:
            last_hidden = out.hidden_states[-1]
            target_texts = [s["target"] for s in raw]
            ph_loss = compute_point_head_loss(
                last_hidden, batch["input_ids"], batch["labels"],
                target_texts, point_head, open_patterns,
                coord_scale=coord_scale,
                max_clicks=max_coord_clicks,
                coord_stats=coord_stats,
            )
            if ph_loss is not None:
                coord_total += float(ph_loss.item())
                n_coord += 1

    model.train()
    if skipped:
        print(f"  [evaluate_loader] skipped {skipped} batches (seq_len > {max_seq_len})")
    return {
        "ce_loss":    ce_total    / n       if n       > 0 else None,
        "coord_loss": coord_total / n_coord if n_coord > 0 else None,
    }


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_json_rows(path: str) -> list[dict]:
    p = Path(path) if os.path.isabs(path) else _REPO / path
    with open(p) as f:
        return json.load(f)


def train_val_test_split(
    rows: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be ≥ 0 and sum to < 1")
    n = len(rows)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test  = int(round(n * test_ratio))
    n_val   = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Split leaves no training rows.")
    return (
        [rows[i] for i in idx[n_test + n_val :]],
        [rows[i] for i in idx[n_test : n_test + n_val]],
        [rows[i] for i in idx[:n_test]],
    )


def _maybe_set_max_pixels(processor, max_pixels: Optional[int]) -> None:
    if max_pixels is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
        print(f"Set image_processor.max_pixels = {max_pixels}")
    else:
        print("Warning: could not set max_pixels on this processor.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data / checkpointing
    p.add_argument("--dataset",    default="data/vcot_dataset_unique.json")
    p.add_argument(
        "--output_dir",
        default="runs/point_head_models",
        help="Run directory: LoRA, processor, point_head.pt, metrics, and checkpoints.",
    )
    p.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to a saved checkpoint directory (LoRA adapter + point_head.pt).",
    )
    p.add_argument(
        "--first_epoch", type=int, default=1,
        help="1-based index of the first epoch in this run.",
    )
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument(
        "--system_prompt", type=str,
        default=(
            "You are a visual chain-of-thought assistant. "
            "For every question, output ONLY a sequence of clicks on the chart regions relevant to answering the question. "
            "Use this exact format: <click>x,y</click><click>x,y</click>... "
            "Coordinates are in a normalized 0-1000 scale where (0,0) is the top-left and (1000,1000) is the bottom-right of the image. "
            "Do not output any text, explanation, or numerical answer. Only output <click> tags."
        ),
    )
    p.add_argument("--val_ratio",  type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--limit",      type=int,   default=None)
    p.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Stop each training epoch after this many batches (smoke tests).",
    )
    p.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Use at most this many validation batches per eval (smoke tests).",
    )
    p.add_argument(
        "--clicks_only",
        action=argparse.BooleanOptionalAction, default=True,
    )

    # Training
    p.add_argument("--epochs",                      type=int,   default=3)
    p.add_argument("--batch_size",                  type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    p.add_argument("--lr",                          type=float, default=2e-4)
    p.add_argument("--weight_decay",                type=float, default=0.01)
    p.add_argument("--warmup_ratio",                type=float, default=0.05)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument("--max_pixels",  type=int,   default=401408)
    p.add_argument("--max_seq_len", type=int,   default=None)

    # LoRA / QLoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--qlora",        action="store_true", default=False)
    p.add_argument("--lora_vision_projector", action="store_true", default=False)
    p.add_argument(
        "--freeze_vision",
        action=argparse.BooleanOptionalAction, default=True,
    )

    # ── PointHead ──────────────────────────────────────────────────────────
    p.add_argument(
        "--coord_lambda", type=float, default=0.5,
        help="Weight λ for the PointHead L1 loss (0 = CE only, disable PointHead).",
    )
    p.add_argument(
        "--coord_scale", type=float, default=1000.0,
        help="Coordinate divisor when --no-normalize_xy (default matches 0-1000 range).",
    )
    p.add_argument(
        "--max_coord_clicks", type=int, default=15,
        help="Max clicks per sample used for PointHead loss (0 = all).",
    )
    p.add_argument(
        "--point_head_dim", type=int, default=256,
        help="Hidden dimension of the PointHead intermediate layer.",
    )
    p.add_argument(
        "--normalize_xy",
        action=argparse.BooleanOptionalAction, default=True,
        help="Normalise x and y independently by dataset std before computing PointHead loss.",
    )

    # Generation eval
    p.add_argument("--gen_eval_epochs",     type=int,   default=1)
    p.add_argument("--gen_eval_steps",      type=int,   default=0)
    p.add_argument("--gen_eval_samples",    type=int,   default=16)
    p.add_argument("--gen_max_new_tokens",  type=int,   default=512)
    p.add_argument("--gen_temperature",     type=float, default=0.3)
    p.add_argument("--gen_repetition_penalty", type=float, default=1.0)
    p.add_argument(
        "--gen_block_repeats",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument("--gen_block_repeat_penalty", type=float, default=50.0)
    p.add_argument("--gen_eval_fixed_slice", action="store_true")
    p.add_argument(
        "--print_samples_every_steps",
        type=int,
        default=100,
        help="Every N optimizer steps, print generations for val samples (0=disable).",
    )
    p.add_argument(
        "--print_samples_n",
        type=int,
        default=2,
        help="How many validation samples to print when print_samples_every_steps fires.",
    )

    p.add_argument(
        "--epoch_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save LoRA adapter + processor + point_head.pt each epoch under epoch_XXXX/.",
    )
    p.add_argument(
        "--log_step_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append one JSON object per optimizer step to step_metrics.jsonl (for loss curves).",
    )
    p.add_argument(
        "--half_epoch_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save under epoch_XXXX_half/ midway through each epoch (always, not val-gated).",
    )
    p.add_argument(
        "--half_epoch_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation when writing a half-epoch checkpoint.",
    )
    p.add_argument(
        "--save_every_batches",
        type=int,
        default=None,
        help="If set (e.g. 2000), save LoRA+processor+point_head under "
        "epoch_XXXX_batch_XXXXXX/ every N dataloader batches within an epoch (no val eval).",
    )
    p.add_argument(
        "--log_training_batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append one JSON line per training batch to training_batches.jsonl.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.first_epoch < 1:
        raise ValueError("--first_epoch must be >= 1")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max_train_batches must be >= 1")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise ValueError("--max_val_batches must be >= 1")
    if args.print_samples_n < 1:
        raise ValueError("--print_samples_n must be >= 1")
    if args.save_every_batches is not None and args.save_every_batches < 1:
        raise ValueError("--save_every_batches must be >= 1 when set")

    resume_path: Optional[Path] = None
    if args.resume_from:
        rp = Path(args.resume_from)
        resume_path = rp.resolve() if rp.is_absolute() else (_REPO / rp).resolve()
        if not (resume_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"--resume_from: missing adapter_config.json under {resume_path}"
            )

    # ── Device & dtype ────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float16
    elif torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.bfloat16
    else:
        device, dtype = torch.device("cpu"), torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Data ──────────────────────────────────────────────────────────────────
    rows = load_json_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]

    train_rows, val_rows, test_rows = train_val_test_split(
        rows, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"Split → train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    coord_stats: Optional[CoordStats] = None
    if args.normalize_xy:
        coord_stats = compute_coord_stats(train_rows, clicks_only=args.clicks_only)
        print(
            f"Coord stats (train): "
            f"x_mean={coord_stats.x_mean:.1f}  x_std={coord_stats.x_std:.1f}  "
            f"y_mean={coord_stats.y_mean:.1f}  y_std={coord_stats.y_std:.1f}"
        )

    train_ds = BubbleViewDataset(train_rows, clicks_only=args.clicks_only)
    val_ds   = BubbleViewDataset(val_rows,   clicks_only=args.clicks_only)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: b)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: b)
    if args.max_val_batches is not None:
        _vb = min(len(val_loader), args.max_val_batches)
        print(
            f"Capping validation to {_vb} batch(es) per eval "
            f"(of {len(val_loader)} in loader)."
        )

    # ── Processor ─────────────────────────────────────────────────────────────
    if resume_path is not None:
        processor = Qwen2VLProcessor.from_pretrained(str(resume_path), use_fast=False)
        print(f"Processor loaded from {resume_path}")
    else:
        processor = Qwen2VLProcessor.from_pretrained(args.model_name, use_fast=False)
    _maybe_set_max_pixels(processor, args.max_pixels)

    # Pre-compute opening `<click>` token runs (several variants; see click_open_patterns).
    open_patterns = click_open_patterns(processor.tokenizer)
    print(
        f"<click> opener pattern(s) ({len(open_patterns)}): "
        f"{[processor.tokenizer.decode(p) for p in open_patterns]}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.qlora and device.type == "cuda":
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        print("Loading model in 4-bit NF4 (QLoRA)…")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name, quantization_config=bnb_cfg, device_map="auto"
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        if args.qlora:
            print(f"Warning: --qlora requires CUDA; using {dtype} LoRA on {device}.")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype
        ).to(device)

    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad_(False)
                frozen += 1
        print(f"Frozen {frozen} ViT backbone tensors.")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    if args.lora_vision_projector:
        target_modules += ["mlp.0", "mlp.2"]
        print("LoRA also targeting visual.merger MLP.")

    effective_dropout = 0.0 if device.type == "mps" else args.lora_dropout
    if device.type == "mps" and args.lora_dropout > 0.0:
        print(f"MPS: overriding lora_dropout → 0.0")

    if resume_path is not None:
        print(f"Resuming LoRA weights from {resume_path}")
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=effective_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if args.gradient_checkpointing and not (args.qlora and device.type == "cuda"):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    # ── PointHead ─────────────────────────────────────────────────────────────
    use_coord = args.coord_lambda > 0.0
    point_head: Optional[PointHead] = None

    if use_coord:
        # Qwen2VLConfig does not expose hidden_size at the top level;
        # read it from the lm_head weight (works with and without PEFT wrapping).
        lm_head_w = model.lm_head.weight   # shape: (vocab_size, hidden_size)
        hidden_size = lm_head_w.shape[1]
        print(f"Detected hidden_size={hidden_size} from lm_head")
        point_head = PointHead(hidden_size, intermediate=args.point_head_dim).to(device)
        # Resume PointHead weights if available
        ph_ckpt = resume_path / "point_head.pt" if resume_path else None
        if ph_ckpt is not None and ph_ckpt.exists():
            point_head.load_state_dict(torch.load(ph_ckpt, map_location=device))
            print(f"PointHead weights loaded from {ph_ckpt}")
        else:
            print(f"PointHead initialised (hidden={hidden_size} → {args.point_head_dim} → 2)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if point_head is not None:
        trainable_params += list(point_head.parameters())

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    accum = max(1, args.gradient_accumulation_steps)
    train_batches_per_epoch = len(train_loader)
    if args.max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, args.max_train_batches)
        print(
            f"Capping training to {train_batches_per_epoch} batch(es) per epoch "
            f"(of {len(train_loader)} in loader)."
        )
    total_steps = math.ceil(train_batches_per_epoch / accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = make_scheduler(opt, warmup_steps, total_steps)
    print(f"Scheduler: {warmup_steps} warmup / {total_steps} total steps.")

    # ── Output dirs / manifests ───────────────────────────────────────────────
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    best_dir = out_root / "best"
    last_dir = out_root / "last"

    with open(out_root / "split_manifest.json", "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                   "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
                   "train_n": len(train_rows), "val_n": len(val_rows),
                   "test_n": len(test_rows)}, f, indent=2)
    with open(out_root / "test_holdout.json", "w") as f:
        json.dump(test_rows, f)

    param_counts = trainable_parameter_counts(model, point_head)
    print(
        "Trainable parameters: "
        + ", ".join(f"{k}={v:,}" for k, v in param_counts.items())
    )

    train_cfg = vars(args).copy()
    if coord_stats is not None:
        train_cfg["coord_stats"] = coord_stats._asdict()
    train_cfg["click_open_patterns"] = open_patterns
    train_cfg["trainable_parameter_counts"] = param_counts
    train_cfg["device"] = str(device)
    train_cfg["dtype"] = str(dtype)
    train_cfg["git_revision"] = _git_revision(_REPO)
    train_cfg["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    train_cfg["run_started_unix"] = run_started
    train_config_path = out_root / "train_config.json"
    with open(train_config_path, "w") as f:
        json.dump(train_cfg, f, indent=2)

    # Load previous best for resume continuity
    best_val: Optional[float] = None
    if resume_path is not None:
        mf = resume_path / "best_metrics.json"
        if mf.is_file():
            try:
                prev = json.load(open(mf))
                vc = prev.get("val_combined")
                if vc is not None and not math.isnan(float(vc)):
                    best_val = float(vc)
                    print(f"Loaded previous best val_combined={best_val:.4f}")
            except Exception:
                pass

    print(f"Active losses: CE" + (f" + PointHead-L1(λ={args.coord_lambda})" if use_coord else ""))
    if args.gen_block_repeats:
        print(f"Inference: ClickRepetitionBlocker (penalty={args.gen_block_repeat_penalty})")

    epoch_last = args.first_epoch + args.epochs - 1

    metrics_history: list[dict] = []
    checkpoint_manifest: list[dict] = []
    step_log_f = (
        open(out_root / "step_metrics.jsonl", "w", encoding="utf-8")
        if args.log_step_metrics
        else None
    )
    batch_log_f = (
        open(out_root / "training_batches.jsonl", "w", encoding="utf-8")
        if args.log_training_batches
        else None
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    global_step   = 0
    grad_norm     = 0.0
    skipped_steps = 0

    for epoch in range(args.epochs):
        epoch_display = args.first_epoch + epoch
        ce_total = coord_total = 0.0
        n_steps = n_coord = 0
        opt.zero_grad(set_to_none=True)
        _cap = args.max_train_batches
        _pbar_total = min(len(train_loader), _cap) if _cap is not None else len(train_loader)
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch_display}/{epoch_last}",
            total=_pbar_total,
        )
        half_batches = (_pbar_total + 1) // 2
        if half_batches >= _pbar_total:
            half_batches = 0
        ce_sum_at_half: Optional[float] = None
        n_steps_at_half = 0
        coord_sum_at_half: Optional[float] = None
        n_coord_at_half = 0

        def _batch_phase(bidx: int) -> str:
            if half_batches == 0:
                return "full"
            return "first_half" if bidx <= half_batches else "second_half"

        for step, raw in enumerate(pbar):
            if _cap is not None and step >= _cap:
                break
            batch = build_batch(raw, processor, device, system_prompt=args.system_prompt)
            seq_len = batch["input_ids"].shape[1]
            if args.max_seq_len is not None and seq_len > args.max_seq_len:
                skipped_steps += 1
                if batch_log_f is not None:
                    batch_log_f.write(
                        json.dumps(
                            {
                                "epoch": epoch_display,
                                "batch_in_epoch": step + 1,
                                "skipped": True,
                                "reason": "max_seq_len",
                                "seq_len": int(seq_len),
                            }
                        )
                        + "\n"
                    )
                    batch_log_f.flush()
                del batch
                if device.type == "mps":
                    torch.mps.empty_cache()
                continue

            # Forward — request hidden states only when PointHead is active
            out = model(**batch, output_hidden_states=use_coord)
            ce_loss = out.loss

            # ── PointHead L1 loss ──────────────────────────────────────────
            coord_val: Optional[torch.Tensor] = None
            if use_coord and out.hidden_states is not None:
                last_hidden  = out.hidden_states[-1]   # (B, T, D)
                target_texts = [s["target"] for s in raw]
                coord_val = compute_point_head_loss(
                    last_hidden, batch["input_ids"], batch["labels"],
                    target_texts, point_head, open_patterns,
                    coord_scale=args.coord_scale,
                    max_clicks=args.max_coord_clicks,
                    coord_stats=coord_stats,
                )

            total_loss = ce_loss
            if coord_val is not None:
                total_loss = total_loss + args.coord_lambda * coord_val

            (total_loss / accum).backward()

            # ── Bookkeeping ───────────────────────────────────────────────
            ce_s    = float(ce_loss.item())
            coord_s = float(coord_val.item()) if coord_val is not None else None
            batch_seq_len = int(batch["input_ids"].shape[1])
            del batch, out, total_loss, ce_loss, coord_val

            ce_total += ce_s
            n_steps  += 1
            if coord_s is not None:
                coord_total += coord_s
                n_coord     += 1

            # ── Gradient accumulation ─────────────────────────────────────
            is_last = (step + 1) == _pbar_total
            if (step + 1) % accum == 0 or is_last:
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                )
                if point_head is not None:
                    has_bad_grad = has_bad_grad or any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in point_head.parameters()
                    )
                if has_bad_grad:
                    skipped_steps += 1
                    opt.zero_grad(set_to_none=True)
                else:
                    all_params = list(model.parameters())
                    if point_head is not None:
                        all_params += list(point_head.parameters())
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    )
                    opt.step()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    if step_log_f is not None:
                        bidx = step + 1
                        step_log_f.write(
                            json.dumps(
                                json_safe_for_metrics(
                                    {
                                        "step": global_step,
                                        "epoch": epoch_display,
                                        "batch_in_epoch": bidx,
                                        "batches_per_epoch": _pbar_total,
                                        "epoch_phase": _batch_phase(bidx),
                                        "ce": ce_s,
                                        "ph_l1": coord_s,
                                        "lr": scheduler.get_last_lr()[0],
                                        "grad_norm": grad_norm,
                                    }
                                )
                            )
                            + "\n"
                        )
                        step_log_f.flush()

                    if (
                        args.print_samples_every_steps > 0
                        and val_rows
                        and global_step % args.print_samples_every_steps == 0
                    ):
                        n_show = min(args.print_samples_n, len(val_rows))
                        if n_show > 0:
                            vstart = (
                                (global_step // args.print_samples_every_steps) * n_show
                            ) % len(val_rows)
                            gen_eval_rows = _gen_eval_slice(val_rows, n_show, vstart)
                            gen_samples = [
                                BubbleViewDataset(gen_eval_rows, args.clicks_only)[i]
                                for i in range(len(gen_eval_rows))
                            ]
                            pbar.write(
                                f"\n── samples @ global_step={global_step} "
                                f"(val[{vstart}:{vstart + n_show}]) ──"
                            )
                            generation_eval(
                                model,
                                gen_samples,
                                processor,
                                device,
                                args.gen_max_new_tokens,
                                system_prompt=args.system_prompt,
                                temperature=args.gen_temperature,
                                repetition_penalty=args.gen_repetition_penalty,
                                block_repeats=args.gen_block_repeats,
                                block_repeat_penalty=args.gen_block_repeat_penalty,
                            )

                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            # ── Mid-training generation eval ──────────────────────────────
            if args.gen_eval_steps > 0 and (step + 1) % args.gen_eval_steps == 0:
                if val_rows:
                    block  = (step + 1) // args.gen_eval_steps
                    vstart = 0 if args.gen_eval_fixed_slice else (block - 1) * args.gen_eval_samples
                    gen_eval_rows = _gen_eval_slice(val_rows, args.gen_eval_samples, vstart)
                    pbar.write(f"\n── step {step+1} gen eval (val[{vstart}..]) ──")
                    gen_samples = [BubbleViewDataset(gen_eval_rows, args.clicks_only)[i]
                                   for i in range(len(gen_eval_rows))]
                    generation_eval(
                        model, gen_samples, processor, device, args.gen_max_new_tokens,
                        system_prompt=args.system_prompt,
                        temperature=args.gen_temperature,
                        repetition_penalty=args.gen_repetition_penalty,
                        block_repeats=args.gen_block_repeats,
                        block_repeat_penalty=args.gen_block_repeat_penalty,
                    )

            # ── Progress bar ──────────────────────────────────────────────
            postfix = {
                "ce":    f"{ce_s:.4f}",
                "gnorm": f"{grad_norm:.2f}",
                "lr":    f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if coord_s is not None:
                postfix["ph_l1"] = f"{coord_s:.4f}"
            if skipped_steps > 0:
                postfix["skip"] = skipped_steps
            pbar.set_postfix(**postfix)

            bidx = step + 1
            if batch_log_f is not None:
                batch_log_f.write(
                    json.dumps(
                        json_safe_for_metrics(
                            {
                                "epoch": epoch_display,
                                "batch_in_epoch": bidx,
                                "batches_per_epoch": _pbar_total,
                                "epoch_phase": _batch_phase(bidx),
                                "ce": ce_s,
                                "ph_l1": coord_s,
                                "seq_len": batch_seq_len,
                            }
                        )
                    )
                    + "\n"
                )
                batch_log_f.flush()

            # ── Half-epoch checkpoint (not gated on validation quality) ─────
            if (
                args.half_epoch_checkpoints
                and half_batches >= 1
                and half_batches < _pbar_total
                and bidx == half_batches
            ):
                train_ce_h = ce_total / max(n_steps, 1)
                train_coord_h = coord_total / n_coord if n_coord > 0 else None
                val_h_ce = val_h_coord = val_h_combined = None
                if args.half_epoch_eval:
                    val_h = evaluate_loader(
                        model,
                        val_loader,
                        processor,
                        device,
                        point_head=point_head,
                        open_patterns=open_patterns,
                        coord_lambda=args.coord_lambda,
                        coord_scale=args.coord_scale,
                        max_coord_clicks=args.max_coord_clicks,
                        system_prompt=args.system_prompt,
                        max_seq_len=args.max_seq_len,
                        coord_stats=coord_stats,
                        max_batches=args.max_val_batches,
                    )
                    val_h_ce = val_h["ce_loss"]
                    val_h_coord = val_h["coord_loss"]
                    val_h_combined = val_h_ce
                    if val_h_combined is not None and val_h_coord is not None:
                        val_h_combined = val_h_combined + args.coord_lambda * val_h_coord

                half_record = json_safe_for_metrics(
                    {
                        "phase": "half",
                        "epoch": epoch_display,
                        "global_step_end": global_step,
                        "train_batches_so_far": n_steps,
                        "train_ce": train_ce_h,
                        "train_ph_l1": train_coord_h,
                        "val_ce": val_h_ce,
                        "val_ph_l1": val_h_coord,
                        "val_combined": val_h_combined,
                        "skipped_optimizer_steps_total": skipped_steps,
                        "trainable_parameter_counts": param_counts,
                    }
                )
                metrics_history.append(half_record)
                with open(out_root / "metrics_history.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)

                half_dir = out_root / f"epoch_{epoch_display:04d}_half"
                save_run_checkpoint(
                    half_dir,
                    model,
                    processor,
                    point_head,
                    half_record,
                    train_config_path,
                )
                checkpoint_manifest.append(
                    {
                        "kind": "half",
                        "epoch": epoch_display,
                        "path": str(half_dir),
                        "metrics": half_record,
                    }
                )
                print(f"  → half-epoch checkpoint → {half_dir}")

                ce_sum_at_half = ce_total
                n_steps_at_half = n_steps
                coord_sum_at_half = coord_total
                n_coord_at_half = n_coord

            # ── Periodic batch checkpoints (no validation; snapshot weights) ───
            if (
                args.save_every_batches
                and args.save_every_batches > 0
                and bidx % args.save_every_batches == 0
            ):
                batch_dir = out_root / f"epoch_{epoch_display:04d}_batch_{bidx:06d}"
                snap = json_safe_for_metrics(
                    {
                        "phase": "batch",
                        "epoch": epoch_display,
                        "batch_in_epoch": bidx,
                        "batches_per_epoch": _pbar_total,
                        "global_step": global_step,
                        "train_batches_so_far": n_steps,
                        "skipped_optimizer_steps_total": skipped_steps,
                    }
                )
                save_run_checkpoint(
                    batch_dir,
                    model,
                    processor,
                    point_head,
                    snap,
                    train_config_path,
                )
                checkpoint_manifest.append(
                    {
                        "kind": "batch",
                        "epoch": epoch_display,
                        "batch_in_epoch": bidx,
                        "path": str(batch_dir),
                        "metrics": snap,
                    }
                )
                print(f"  → batch checkpoint → {batch_dir}")

        # ── Epoch-end validation ──────────────────────────────────────────────
        if skipped_steps > 0:
            print(f"  ⚠ {skipped_steps} optimizer steps skipped (NaN/Inf grads).")

        train_ce    = ce_total    / max(n_steps, 1)
        train_coord = coord_total / n_coord if n_coord > 0 else None

        val_metrics = evaluate_loader(
            model, val_loader, processor, device,
            point_head=point_head, open_patterns=open_patterns,
            coord_lambda=args.coord_lambda, coord_scale=args.coord_scale,
            max_coord_clicks=args.max_coord_clicks,
            system_prompt=args.system_prompt,
            max_seq_len=args.max_seq_len,
            coord_stats=coord_stats,
            max_batches=args.max_val_batches,
        )
        val_ce    = val_metrics["ce_loss"]
        val_coord = val_metrics["coord_loss"]

        val_combined = val_ce
        if val_combined is not None and val_coord is not None:
            val_combined += args.coord_lambda * val_coord

        # ── Generation eval ───────────────────────────────────────────────────
        gen_metrics: dict = {}
        if args.gen_eval_epochs > 0 and (epoch + 1) % args.gen_eval_epochs == 0 and val_rows:
            vstart = 0 if args.gen_eval_fixed_slice else (
                ((args.first_epoch - 1 + epoch) * args.gen_eval_samples) % len(val_rows)
            )
            gen_eval_rows = _gen_eval_slice(val_rows, args.gen_eval_samples, vstart)
            gen_samples = [BubbleViewDataset(gen_eval_rows, args.clicks_only)[i]
                           for i in range(len(gen_eval_rows))]
            gen_metrics = generation_eval(
                model, gen_samples, processor, device, args.gen_max_new_tokens,
                system_prompt=args.system_prompt,
                temperature=args.gen_temperature,
                repetition_penalty=args.gen_repetition_penalty,
                block_repeats=args.gen_block_repeats,
                block_repeat_penalty=args.gen_block_repeat_penalty,
            )

        # ── Logging ───────────────────────────────────────────────────────────
        parts = [
            f"epoch {epoch_display}",
            f"train_ce={train_ce:.4f}",
            f"val_ce={val_ce:.4f}" if val_ce is not None else "val_ce=—",
        ]
        if train_coord is not None: parts.append(f"train_ph={train_coord:.4f}")
        if val_coord   is not None: parts.append(f"val_ph={val_coord:.4f}")
        if gen_metrics:
            parts.append(f"gen_dtw={gen_metrics['gen_dtw']:.2f}px")
            parts.append(f"gen_len_err={gen_metrics['gen_len_err']:.3f}")
        print("  ".join(parts))

        new_best = False
        if val_combined is not None and not math.isnan(val_combined):
            if best_val is None or val_combined < best_val:
                new_best = True
                best_val = val_combined
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                if point_head is not None:
                    torch.save(point_head.state_dict(), best_dir / "point_head.pt")
                with open(best_dir / "best_metrics.json", "w") as f:
                    json.dump(
                        json_safe_for_metrics(
                            {
                                "epoch": epoch_display,
                                "val_ce": val_ce,
                                "val_ph_l1": val_coord,
                                "val_combined": val_combined,
                                "train_ce": train_ce,
                                "train_ph_l1": train_coord,
                                **gen_metrics,
                            }
                        ),
                        f,
                        indent=2,
                    )
                print(f"  → best checkpoint saved (val_combined={val_combined:.4f})")

        train_ce_first_half = train_ph_first_half = None
        train_ce_second_half = train_ph_second_half = None
        if ce_sum_at_half is not None and n_steps_at_half > 0:
            train_ce_first_half = ce_sum_at_half / n_steps_at_half
            if n_coord_at_half > 0 and coord_sum_at_half is not None:
                train_ph_first_half = coord_sum_at_half / n_coord_at_half
        if ce_sum_at_half is not None and n_steps > n_steps_at_half:
            train_ce_second_half = (ce_total - ce_sum_at_half) / max(
                n_steps - n_steps_at_half, 1
            )
            dn_coord = n_coord - n_coord_at_half
            if dn_coord > 0 and coord_sum_at_half is not None:
                train_ph_second_half = (coord_total - coord_sum_at_half) / dn_coord

        epoch_record = json_safe_for_metrics(
            {
                "phase": "end",
                "epoch": epoch_display,
                "batches_per_epoch": _pbar_total,
                "global_step_end": global_step,
                "train_ce": train_ce,
                "train_ph_l1": train_coord,
                "train_ce_first_half": train_ce_first_half,
                "train_ph_first_half": train_ph_first_half,
                "train_ce_second_half": train_ce_second_half,
                "train_ph_second_half": train_ph_second_half,
                "val_ce": val_ce,
                "val_ph_l1": val_coord,
                "val_combined": val_combined,
                "new_best_val_combined": new_best,
                "skipped_optimizer_steps_total": skipped_steps,
                "trainable_parameter_counts": param_counts,
                **gen_metrics,
            }
        )
        metrics_history.append(epoch_record)
        with open(out_root / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)

        if args.epoch_checkpoints:
            ep_dir = out_root / f"epoch_{epoch_display:04d}"
            save_run_checkpoint(
                ep_dir,
                model,
                processor,
                point_head,
                epoch_record,
                train_config_path,
            )
            checkpoint_manifest.append(
                {
                    "kind": "end",
                    "epoch": epoch_display,
                    "path": str(ep_dir),
                    "metrics": epoch_record,
                }
            )
            print(f"  → epoch checkpoint → {ep_dir}")

    if step_log_f is not None:
        step_log_f.close()
    if batch_log_f is not None:
        batch_log_f.close()

    train_cfg["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    train_cfg["duration_wall_seconds"] = time.time() - run_started
    train_cfg["final_global_step"] = global_step
    train_cfg["final_best_val_combined"] = best_val
    with open(train_config_path, "w") as f:
        json.dump(train_cfg, f, indent=2)

    report_doc = json_safe_for_metrics(
        {
            "run_directory": str(out_root.resolve()),
            "train_config": str(train_config_path.resolve()),
            "metrics_history": str((out_root / "metrics_history.json").resolve()),
            "step_metrics_jsonl": str((out_root / "step_metrics.jsonl").resolve())
            if args.log_step_metrics
            else None,
            "training_batches_jsonl": str((out_root / "training_batches.jsonl").resolve())
            if args.log_training_batches
            else None,
            "split_manifest": str((out_root / "split_manifest.json").resolve()),
            "test_holdout": str((out_root / "test_holdout.json").resolve()),
            "checkpoints": checkpoint_manifest,
            "last_checkpoint": str(last_dir.resolve()),
            "best_checkpoint_dir": str(best_dir.resolve()),
            "best_val_combined": best_val,
            "git_revision": train_cfg.get("git_revision"),
            "duration_wall_seconds": train_cfg["duration_wall_seconds"],
        }
    )
    with open(out_root / "report.json", "w") as f:
        json.dump(report_doc, f, indent=2)
    print(f"Report manifest → {out_root / 'report.json'}")

    # ── Save last epoch ───────────────────────────────────────────────────────
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    if point_head is not None:
        torch.save(point_head.state_dict(), last_dir / "point_head.pt")
    print(f"Last epoch saved → {last_dir}")
    if best_val is not None:
        print(f"Best val_combined = {best_val:.4f}  → {best_dir}")


if __name__ == "__main__":
    main()
