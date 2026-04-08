from __future__ import annotations

import argparse
import json
import math
import os
import re
import random
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

_REPO = Path(__file__).resolve().parent.parent
_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")

# Assistant turn delimiter used by Qwen chat template
_ASST_HEADER = "<|im_start|>assistant"


def dtw_distance(seq_a: list[tuple[float, float]],
                 seq_b: list[tuple[float, float]]) -> float:

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
            cost = dist(seq_a[i - 1], seq_b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Normalise by path length so short/long sequences are comparable
    return dp[n][m] / (n + m)


def parse_clicks(text: str) -> list[tuple[float, float]]:
    """Extract (x, y) pairs from a click-sequence string."""
    return [(float(x), float(y)) for x, y in _RE_CLICK.findall(text)]


def soft_dtw(
    pred: torch.Tensor,    # (N, 2)
    target: torch.Tensor,  # (M, 2)
    gamma: float = 1.0,
) -> torch.Tensor:

    n, m = pred.shape[0], target.shape[0]
    if n == 0 or m == 0:
        return pred.sum() * 0.0

    # MPS-safe pairwise distance
    diff = pred.unsqueeze(1) - target.unsqueeze(0)   # (N, M, 2)
    cost = diff.pow(2).sum(-1).clamp(min=1e-12).sqrt()  # (N, M)

    INF = torch.tensor(1e9, dtype=pred.dtype, device=pred.device)
    ZERO = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

    # use list-of-lists to keep autograd graph intact
    R = [[INF] * (m + 1) for _ in range(n + 1)]
    R[0][0] = ZERO

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            neighbors = torch.stack([R[i-1][j], R[i][j-1], R[i-1][j-1]])
            soft_min = -gamma * torch.logsumexp(-neighbors / gamma, dim=0)
            R[i][j] = cost[i-1, j-1] + soft_min

    return R[n][m] / (n * m)



def build_int_token_map(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:

    ids: list[int] = []
    vals: list[float] = []
    for tok_str, tok_id in tokenizer.get_vocab().items():
        decoded = tokenizer.convert_tokens_to_string([tok_str]).strip()
        if decoded and decoded.isdecimal() and decoded.isascii():
            ids.append(tok_id)
            vals.append(float(decoded))
    if not ids:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable expected-coordinate extraction
# ─────────────────────────────────────────────────────────────────────────────

def _find_subseq(
    seq: torch.Tensor, pattern: list[int], start: int = 0
) -> Optional[int]:
    m = len(pattern)
    if m == 0:
        return start
    pat = torch.tensor(pattern, device=seq.device, dtype=seq.dtype)
    for i in range(start, len(seq) - m + 1):
        if (seq[i : i + m] == pat).all():
            return i
    return None


def _expected_coord(
    logits: torch.Tensor,     
    positions: torch.Tensor,  
    tok_strs: list[str],      
    int_ids: torch.Tensor,   
    int_vals: torch.Tensor,   
) -> torch.Tensor:

    suffix_len = sum(len(s) for s in tok_strs)

    result = logits.new_zeros(()).float()
    for s, pos in zip(tok_strs, positions.tolist()):
        suffix_len -= len(s)
        weight = 10.0 ** suffix_len

        full_probs = F.softmax(logits[pos].float(), dim=-1)
        int_probs  = full_probs[int_ids]
        ev = (int_probs * int_vals).sum()
        result = result + weight * ev
    return result


def extract_expected_click_seqs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_text: str,
    tokenizer,
    int_ids: torch.Tensor,
    int_vals: torch.Tensor,
    max_clicks: int = 0,    
) -> tuple[Optional[torch.Tensor], list[tuple[float, float]]]:

    clicks = _RE_CLICK.findall(target_text)
    if not clicks:
        return None, []
    if max_clicks > 0:
        clicks = clicks[:max_clicks]

    asst_pos = (labels != -100).nonzero(as_tuple=True)[0]
    if len(asst_pos) == 0:
        return None, []
    asst_ids = input_ids[asst_pos]

    int_ids_d  = int_ids.to(logits.device)
    int_vals_d = int_vals.to(logits.device, dtype=logits.dtype)

    pred_list: list[torch.Tensor] = []
    gt_list:   list[tuple[float, float]] = []
    skipped = 0
    search_off = 0

    for x_str, y_str in clicks:
        x_toks = tokenizer.encode(x_str, add_special_tokens=False)
        y_toks = tokenizer.encode(y_str, add_special_tokens=False)

        x_start = _find_subseq(asst_ids, x_toks, search_off)
        if x_start is None:
            skipped += 1
            continue
        x_seq_pos = asst_pos[x_start : x_start + len(x_toks)]
        x_tok_strs = [tokenizer.decode([t]).strip() for t in x_toks]
    

        y_start = _find_subseq(asst_ids, y_toks, x_start + len(x_toks) + 1)
        if y_start is None:
            skipped += 1
            continue
        y_seq_pos = asst_pos[y_start : y_start + len(y_toks)]
        y_tok_strs = [tokenizer.decode([t]).strip() for t in y_toks]
    
        pred_x = _expected_coord(logits, x_seq_pos, x_tok_strs, int_ids_d, int_vals_d)
        pred_y = _expected_coord(logits, y_seq_pos, y_tok_strs, int_ids_d, int_vals_d)
        pred_list.append(torch.stack([pred_x, pred_y]))
        gt_list.append((float(x_str), float(y_str)))
        search_off = y_start + len(y_toks)
        
    
    if skipped > 0:
        warnings.warn(
            f"extract_expected_click_seqs: {skipped}/{len(clicks)} clicks "
            "could not be located in the assistant token sequence. "
            "This may indicate a tokenisation mismatch.",
            stacklevel=2,
        )

    if not pred_list:
        return None, []
    return torch.stack(pred_list), gt_list


def compute_batch_sequence_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_texts: list[str],
    tokenizer,
    int_ids: torch.Tensor,
    int_vals: torch.Tensor,
    coord_scale: float = 1000.0,
    use_sdtw: bool = True,
    sdtw_gamma: float = 1.0,
    max_coord_clicks: int = 0,
) -> Optional[torch.Tensor]:

    losses: list[torch.Tensor] = []

    for i, tgt in enumerate(target_texts):
       
        pred_coords, gt_pairs = extract_expected_click_seqs(
            logits[i], input_ids[i], labels[i], tgt, tokenizer, int_ids, int_vals,
            max_clicks=max_coord_clicks,
        )
        if pred_coords is None or not gt_pairs:
            continue

        gt_t = torch.tensor(
            gt_pairs, dtype=pred_coords.dtype, device=pred_coords.device
        ) / coord_scale
        pred_s = pred_coords / coord_scale

    
        if use_sdtw:
            dtw_loss = soft_dtw(pred_s, gt_t, gamma=sdtw_gamma)
            
            losses.append(dtw_loss)
        else:
            n = min(len(pred_s), len(gt_t))
            diff = pred_s[:n] - gt_t[:n]
            losses.append(diff.pow(2).sum(-1).add(1e-12).sqrt().mean())

    return torch.stack(losses).mean() if losses else None


_SEP = " | <sep> |"


class BubbleViewDataset(Dataset):
    def __init__(self, rows: list[dict], clicks_only: bool = True):
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


def _find_assistant_start(input_ids: torch.Tensor, tokenizer) -> int:
 
    header_ids = tokenizer.encode(_ASST_HEADER, add_special_tokens=False)
    n = len(input_ids)
    m = len(header_ids)
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
    samples: list[dict], processor: Qwen2VLProcessor, device: torch.device,
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
                    {"type": "text", "text": s["prompt"]},
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

    batch = processor(
        text=full_texts, images=images, return_tensors="pt", padding=True
    )
    input_ids = batch["input_ids"]        # (B, L)
    labels = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id

    for i in range(len(samples)):
        asst_start = _find_assistant_start(input_ids[i], processor.tokenizer)
        labels[i, :asst_start] = -100   # mask prompt + image tokens

    if pad_id is not None:
        labels[labels == pad_id] = -100

    batch["labels"] = labels
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _gen_eval_slice(rows: list[dict], n: int, start: int) -> list[dict]:
    """Take n validation rows starting at start, wrapping at end (so we see different charts each eval)."""
    if not rows or n <= 0:
        return []
    L = len(rows)
    start = start % L
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
) -> dict[str, float]:

    model.eval()
    dtw_scores: list[float] = []
    len_errors: list[float] = []
    per_sample: list[dict] = []

    for i, s in enumerate(samples):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text", "text": s["prompt"]},
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
        else:
            gen_kw["do_sample"] = False
        out_ids = model.generate(**inputs, **gen_kw)
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        generated = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_clicks = parse_clicks(generated)
        gt_clicks   = parse_clicks(s["target"])
        score = dtw_distance(pred_clicks, gt_clicks)
        dtw_scores.append(score)

        gt_len = len(gt_clicks)
        pred_len = len(pred_clicks)
        len_err = abs(pred_len - gt_len) / max(gt_len, 1)
        len_errors.append(len_err)

        per_sample.append({
            "idx":         i,
            "image":       s["image"],
            "prompt":      s["prompt"],
            "gt_clicks":   gt_clicks,
            "pred_clicks": pred_clicks,
            "dtw":         score,
            "len_err":     len_err,
        })

        if verbose:
            img_w = s.get("img_w", 0)
            img_h = s.get("img_h", 0)
            img_label = s["image"] if isinstance(s["image"], str) else f"PIL({img_w}x{img_h})"
            print(f"\n[gen_eval {i+1}/{len(samples)}] {img_label}  (img {img_w}x{img_h})")
            print(f"  Prompt  : {s['prompt']}")
            print(f"  Raw out : {generated[:200]!r}")
            print(f"  GT  (norm 0-1000, {gt_len:2d} clicks): {gt_clicks}")
            if gt_clicks and img_w > 0 and img_h > 0:
                gt_px = [(round(x / 1000 * img_w, 1), round(y / 1000 * img_h, 1)) for x, y in gt_clicks]
                print(f"  GT  (pixel space ): {gt_px}")
            print(f"  Pred (norm 0-1000, {pred_len:2d} clicks): {pred_clicks}")
            if pred_clicks and img_w > 0 and img_h > 0:
                pred_px = [(round(x / 1000 * img_w, 1), round(y / 1000 * img_h, 1)) for x, y in pred_clicks]
                print(f"  Pred (pixel space ): {pred_px}")
            print(f"  DTW={score:.2f} (1000-scale)  len_err={len_err:.3f}")

    model.train()
    return {
        "gen_dtw":      sum(dtw_scores) / len(dtw_scores) if dtw_scores else float("nan"),
        "gen_len_err":  sum(len_errors) / len(len_errors) if len_errors else float("nan"),
        "per_sample":   per_sample,
    }



@torch.no_grad()
def evaluate_loader(
    model,
    loader: DataLoader,
    processor: Qwen2VLProcessor,
    device: torch.device,
    int_ids: Optional[torch.Tensor],
    int_vals: Optional[torch.Tensor],
    coord_lambda: float,
    coord_scale: float,
    use_sdtw: bool,
    sdtw_gamma: float,
    max_coord_clicks: int = 0,
    system_prompt: str = "",
) -> dict[str, float | None]:
    if len(loader) == 0:
        return {"ce_loss": None, "coord_loss": None}

    model.eval()
    ce_total = coord_total = 0.0
    n = n_coord = 0
    use_coord = coord_lambda > 0 and int_ids is not None

    for raw in loader:
        batch = build_batch(raw, processor, device, system_prompt=system_prompt)
        out = model(**batch)
        ce_total += float(out.loss.item())
        n += 1

        if use_coord:
            target_texts = [s["target"] for s in raw]
            coord_val = compute_batch_sequence_loss(
                out.logits, batch["input_ids"], batch["labels"],
                target_texts, processor.tokenizer, int_ids, int_vals,
                coord_scale=coord_scale, use_sdtw=use_sdtw, sdtw_gamma=sdtw_gamma,
                max_coord_clicks=max_coord_clicks,
            )
            if coord_val is not None:
                coord_total += float(coord_val.item())
                n_coord += 1

    model.train()
    return {
        "ce_loss":    ce_total / n,
        "coord_loss": coord_total / n_coord if n_coord > 0 else None,
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to 0."""
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
        raise ValueError("Split leaves no training rows; reduce ratios or use more data.")
    return (
        [rows[i] for i in idx[n_test + n_val :]],
        [rows[i] for i in idx[n_test : n_test + n_val]],
        [rows[i] for i in idx[:n_test]],
    )


def _maybe_set_max_pixels(processor, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
        print(f"Set image_processor.max_pixels = {max_pixels}")
    else:
        print("Warning: could not set max_pixels on this processor.")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data / checkpointing
    p.add_argument("--dataset",    default="data/vcot_dataset_unique.json")
    p.add_argument("--output_dir", default="runs/qwen_lora")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a visual chain-of-thought assistant. "
            "For every question, output ONLY a sequence of clicks on the chart regions relevant to answering the question. "
            "Use this exact format: <click>x,y</click><click>x,y</click>... "
            "Coordinates are in a normalized 0-1000 scale where (0,0) is the top-left and (1000,1000) is the bottom-right of the image. "
            "Do not output any text, explanation, or numerical answer. Only output <click> tags."
        ),
        help="System message prepended to every conversation. Pass empty string to disable.",
    )
    p.add_argument("--val_ratio",  type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--limit",      type=int,   default=None,
                   help="Use only first N rows (debug).")
    p.add_argument(
        "--clicks_only",
        action=argparse.BooleanOptionalAction, default=True,
        help="Strip '| <sep> | ANSWER' suffix — train on click sequence only.",
    )

    # Training
    p.add_argument("--epochs",                    type=int,   default=3)
    p.add_argument("--batch_size",                type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr",                        type=float, default=2e-4)
    p.add_argument("--weight_decay",              type=float, default=0.01)
    p.add_argument("--warmup_ratio",              type=float, default=0.05,
                   help="Fraction of total steps used for LR warmup.")
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument("--max_pixels", type=int, default=401408)

    # LoRA / QLoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--qlora", action="store_true", default=False)
    p.add_argument("--lora_vision_projector", action="store_true", default=False)
    p.add_argument(
        "--freeze_vision",
        action=argparse.BooleanOptionalAction, default=True,
    )


    p.add_argument("--coord_lambda", type=float, default=0.1,
                   help="Weight λ for spatial auxiliary loss (0 = CE only).")
    p.add_argument("--coord_scale",  type=float, default=1000.0,
                   help="Divide coordinates by this before computing distances.")
    p.add_argument("--max_coord_clicks", type=int, default=15,
                   help="Max clicks per sample for coord loss (0 = no limit). "
                        "Caps expensive samples to avoid slow steps.")
    p.add_argument(
        "--use_sdtw", action="store_true", default=True,
        help="Use soft-DTW instead of pointwise Euclidean. "
             "Handles variable sequence lengths correctly.",
    )
    p.add_argument("--sdtw_gamma", type=float, default=1.0,
                   help="Smoothing factor for soft-DTW.")

    # Generation eval
    p.add_argument("--gen_eval_epochs", type=int, default=1,
                   help="Run generation-based (DTW) eval every N epochs. 0 = never.")
    p.add_argument("--gen_eval_steps", type=int, default=0,
                   help="Also run generation eval every N optimizer steps during training. 0 = never.")
    p.add_argument("--gen_eval_samples", type=int, default=16,
                   help="Number of val samples to use for generation eval.")
    p.add_argument("--gen_max_new_tokens", type=int, default=512)
    p.add_argument(
        "--gen_temperature", type=float, default=0.0,
        help="Generation temperature. 0 = greedy (deterministic). >0 enables sampling.",
    )
    p.add_argument(
        "--gen_eval_fixed_slice", action="store_true",
        help="If set, always use the first --gen_eval_samples val rows (no rotation).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    print(
        f"Split → train={len(train_rows)}  val={len(val_rows)}  "
        f"test={len(test_rows)}  (seed={args.seed})"
    )

    train_ds = BubbleViewDataset(train_rows, clicks_only=args.clicks_only)
    val_ds   = BubbleViewDataset(val_rows,   clicks_only=args.clicks_only)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=lambda b: b
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b
    )

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = Qwen2VLProcessor.from_pretrained(args.model_name)
    _maybe_set_max_pixels(processor, args.max_pixels)

    int_ids, int_vals = build_int_token_map(processor.tokenizer)
    print(f"Integer-valued vocabulary tokens: {len(int_ids)}")

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
            print(f"Warning: --qlora requires CUDA; using fp{dtype} LoRA on {device}.")
        print(f"Loading model in {dtype}…")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype
        ).to(device)

    # ── Freeze ViT backbone (before get_peft_model) ───────────────────────────
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad_(False)
                frozen += 1
        print(f"Frozen {frozen} ViT backbone tensors (visual.* except merger).")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if args.lora_vision_projector:
        target_modules += ["mlp.0", "mlp.2"]
        print("LoRA also targeting visual.merger MLP (mlp.0, mlp.2).")

    # MPS backward pass is unreliable with dropout; force it off on MPS.
    effective_dropout = 0.0 if device.type == "mps" else args.lora_dropout
    if device.type == "mps" and args.lora_dropout > 0.0:
        print(f"MPS detected: overriding lora_dropout {args.lora_dropout} → 0.0 "
              "(MPS dropout backward is unreliable).")

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

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    accum = max(1, args.gradient_accumulation_steps)
    total_steps   = math.ceil(len(train_loader) / accum) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler = make_scheduler(opt, warmup_steps, total_steps)
    print(f"Scheduler: {warmup_steps} warmup steps / {total_steps} total steps.")

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "split_manifest.json", "w") as f:
        json.dump({
            "dataset": args.dataset, "seed": args.seed,
            "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
            "train_n": len(train_rows), "val_n": len(val_rows), "test_n": len(test_rows),
        }, f, indent=2)

    with open(out_root / "test_holdout.json", "w") as f:
        json.dump(test_rows, f)

    with open(out_root / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Held-out test set: {out_root / 'test_holdout.json'} ({len(test_rows)} samples)")

    best_dir = out_root / "best"
    last_dir = out_root / "last"
    best_val: float | None = None
    use_coord = args.coord_lambda > 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    grad_norm: float = 0.0
    skipped_steps: int = 0

    for epoch in range(args.epochs):
        ce_total = coord_total = 0.0
        n_steps = n_coord = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for step, raw in enumerate(pbar):
            batch = build_batch(raw, processor, device, system_prompt=args.system_prompt)
            out   = model(**batch)
            ce_loss = out.loss

            coord_val: Optional[torch.Tensor] = None
            if use_coord:
                target_texts = [s["target"] for s in raw]
                coord_val = compute_batch_sequence_loss(
                    out.logits, batch["input_ids"], batch["labels"],
                    target_texts, processor.tokenizer, int_ids, int_vals,
                    coord_scale=args.coord_scale,
                    use_sdtw=args.use_sdtw,
                    sdtw_gamma=args.sdtw_gamma,
                    max_coord_clicks=args.max_coord_clicks,
                )
            
            del out  # free logits before backward

            total_loss = ce_loss if coord_val is None else ce_loss + args.coord_lambda * coord_val
            (total_loss / accum).backward()

            ce_s    = float(ce_loss.item())
            coord_s = float(coord_val.item()) if coord_val is not None else None
            del batch, total_loss, ce_loss, coord_val

            ce_total += ce_s
            if coord_s is not None:
                coord_total += coord_s
                n_coord += 1
            n_steps += 1

            # Gradient accumulation
            is_last = step == len(train_loader) - 1
            if (step + 1) % accum == 0 or is_last:
                # Detect NaN/Inf gradients and skip the step to protect Adam buffers
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                )
                if has_bad_grad:
                    skipped_steps += 1
                    opt.zero_grad(set_to_none=True)
                    if device.type == "mps":
                        torch.mps.empty_cache()
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    ))
                    opt.step()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1

                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            # Mid-training generation eval every gen_eval_steps training steps
            if args.gen_eval_steps > 0 and (step + 1) % args.gen_eval_steps == 0:
                if val_rows:
                    if args.gen_eval_fixed_slice:
                        vstart = 0
                    else:
                        block = (step + 1) // args.gen_eval_steps
                        vstart = (block - 1) * args.gen_eval_samples
                    gen_eval_rows = _gen_eval_slice(val_rows, args.gen_eval_samples, vstart)
                    pbar.write(
                        f"\n── training step {step + 1} gen eval "
                        f"(val index {vstart}..{vstart + len(gen_eval_rows) - 1} mod {len(val_rows)}) ──"
                    )
                    gen_samples = [BubbleViewDataset(gen_eval_rows, clicks_only=args.clicks_only)[i]
                                     for i in range(len(gen_eval_rows))]
                    generation_eval(
                        model, gen_samples, processor, device, args.gen_max_new_tokens,
                        system_prompt=args.system_prompt,
                        temperature=args.gen_temperature,
                    )

            postfix: dict = {
                "ce": f"{ce_s:.4f}",
                "gnorm": f"{grad_norm:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if coord_s is not None:
                postfix["coord"] = f"{coord_s:.4f}"
            if skipped_steps > 0:
                postfix["skip"] = skipped_steps
            pbar.set_postfix(**postfix)

        # ── Epoch-end metrics ─────────────────────────────────────────────────
        if skipped_steps > 0:
            print(f"  ⚠ {skipped_steps} optimizer steps skipped due to NaN/Inf gradients.")
        train_ce    = ce_total / max(n_steps, 1)
        train_coord = coord_total / n_coord if n_coord > 0 else None

        val_metrics = evaluate_loader(
            model, val_loader, processor, device,
            int_ids=int_ids if use_coord else None,
            int_vals=int_vals if use_coord else None,
            coord_lambda=args.coord_lambda,
            coord_scale=args.coord_scale,
            use_sdtw=args.use_sdtw,
            sdtw_gamma=args.sdtw_gamma,
            max_coord_clicks=args.max_coord_clicks,
            system_prompt=args.system_prompt,
        )
        val_ce    = val_metrics["ce_loss"]
        val_coord = val_metrics["coord_loss"]

        # Combined metric for checkpoint selection
        if val_ce is not None and val_coord is not None:
            val_combined = val_ce + args.coord_lambda * val_coord
        else:
            val_combined = val_ce

        # Generation-based eval (every gen_eval_epochs epochs)
        gen_metrics: dict = {}
        if args.gen_eval_epochs > 0 and (epoch + 1) % args.gen_eval_epochs == 0:
            if val_rows:
                vstart = 0 if args.gen_eval_fixed_slice else (
                    (epoch * args.gen_eval_samples) % len(val_rows)
                )
                gen_eval_rows = _gen_eval_slice(val_rows, args.gen_eval_samples, vstart)
                print(f"  gen_eval val slice start={vstart} (epoch {epoch + 1})")
                gen_samples = [BubbleViewDataset(gen_eval_rows, clicks_only=args.clicks_only)[i]
                               for i in range(len(gen_eval_rows))]
                gen_metrics = generation_eval(
                    model, gen_samples, processor, device, args.gen_max_new_tokens,
                    system_prompt=args.system_prompt,
                    temperature=args.gen_temperature,
                )

        # ── Logging ───────────────────────────────────────────────────────────
        parts = [
            f"epoch {epoch + 1}",
            f"train_ce={train_ce:.4f}",
            f"val_ce={val_ce:.4f}" if val_ce is not None else "val_ce=—",
        ]
        if train_coord is not None: parts.append(f"train_coord={train_coord:.4f}")
        if val_coord   is not None: parts.append(f"val_coord={val_coord:.4f}")
        if gen_metrics:
            parts.append(f"gen_dtw={gen_metrics['gen_dtw']:.2f}px")
            parts.append(f"gen_len_err={gen_metrics['gen_len_err']:.3f}")
        print("  ".join(parts))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_combined is not None and not math.isnan(val_combined):
            if best_val is None or val_combined < best_val:
                best_val = val_combined
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                with open(best_dir / "best_metrics.json", "w") as f:
                    json.dump({
                        "epoch": epoch + 1,
                        "val_ce": val_ce, "val_coord": val_coord,
                        "val_combined": val_combined,
                        "train_ce": train_ce, "train_coord": train_coord,
                        **gen_metrics,
                    }, f, indent=2)
                print(f"  → best checkpoint saved (val_combined={val_combined:.4f})")

    # ── Save last epoch ───────────────────────────────────────────────────────
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    print(f"Last epoch saved → {last_dir}")
    if best_val is not None:
        print(f"Best val_combined = {best_val:.4f}  → {best_dir}")


if __name__ == "__main__":
    main()