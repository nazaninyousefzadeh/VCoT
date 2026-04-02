
from __future__ import annotations

import argparse
import json
import math
import os
import re
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

_REPO = Path(__file__).resolve().parent.parent
_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")


# ─────────────────────────────────────────────────────────────────────────────
# Soft-DTW
# ─────────────────────────────────────────────────────────────────────────────

def _soft_dtw_on_cost(D: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Soft-DTW DP on a pairwise cost matrix D of shape (m, n).
    Recurrence (1-indexed R of shape (m+1, n+1)):
        R[i,j] = D[i-1,j-1] + softmin(R[i-1,j-1], R[i-1,j], R[i,j-1])
    softmin uses the numerically-stable log-sum-exp identity.
    Returns a differentiable scalar.
    """
    m, n = D.shape
    R = D.new_full((m + 1, n + 1), float("inf"))
    R[0, 0] = D.new_zeros(())

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cands = torch.stack([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]])
            # Stable soft-min: -γ log Σ exp(-c_k / γ)
            cmin = cands.min()
            softmin = -gamma * torch.log(torch.exp(-(cands - cmin) / gamma).sum()) - cmin
            R[i, j] = D[i - 1, j - 1] + softmin

    return R[m, n]


def soft_dtw_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Soft-DTW between two sequences of 2-D points.

    Args:
        pred   : (m, 2) predicted (x, y) pairs — must be differentiable.
        target : (n, 2) ground-truth (x, y) pairs.
        gamma  : smoothness; larger → softer (less sensitive to outlier paths).
        normalize: divide by (m + n) so loss is independent of sequence length.

    Returns a scalar tensor.  When either sequence is empty, returns 0.
    """
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.sum() * 0.0  # zero, but keeps gradient graph intact

    # Euclidean pairwise distance matrix (m, n), differentiable w.r.t. pred.
    # Avoids torch.cdist whose backward is not implemented on MPS.
    diff = pred.unsqueeze(1) - target.unsqueeze(0)   # (m, n, 2)
    D = diff.pow(2).sum(-1).add(1e-12).sqrt()         # (m, n)
    val = _soft_dtw_on_cost(D, gamma)
    if normalize:
        val = val / (pred.shape[0] + target.shape[0])
    # Soft-DTW can be negative due to the entropy regularisation term; clamp so
    # it never reduces the total loss (it should act as a penalty, not a reward).
    return val.clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Integer-token helpers (expected coordinate extraction)
# ─────────────────────────────────────────────────────────────────────────────

def build_int_token_map(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scan the vocabulary for tokens whose decoded text is a pure non-negative
    integer (e.g. "0", "25", "546").

    Returns:
        ids  : (K,) long tensor of token IDs
        vals : (K,) float tensor of their integer values
    """
    ids: list[int] = []
    vals: list[float] = []
    for tok_str, tok_id in tokenizer.get_vocab().items():
        decoded = tokenizer.convert_tokens_to_string([tok_str]).strip()
        # isdecimal() + isascii() restricts to plain "0"-"9" digit strings;
        # isdigit() would also match Unicode subscript/superscript chars like ₀
        # which cannot be converted to int and cause a ValueError.
        if decoded and decoded.isdecimal() and decoded.isascii():
            ids.append(tok_id)
            vals.append(float(decoded))
    if not ids:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
    )


def _find_subseq(seq: torch.Tensor, pattern: list[int], start: int = 0) -> Optional[int]:
    """
    First index i >= start such that seq[i : i+len(pattern)] == pattern.
    Returns None if not found.
    """
    m = len(pattern)
    if m == 0:
        return start
    pat = torch.tensor(pattern, device=seq.device, dtype=seq.dtype)
    for i in range(start, len(seq) - m + 1):
        if (seq[i : i + m] == pat).all():
            return i
    return None


def _expected_coord(
    logits: torch.Tensor,     # (seq_len, vocab_size)
    positions: torch.Tensor,  # (P,) indices into seq_len
    tok_strs: list[str],      # decoded string for each token in positions
    int_ids: torch.Tensor,    # (K,) on same device as logits
    int_vals: torch.Tensor,   # (K,) float values
) -> torch.Tensor:
    """
    Differentiable expected coordinate value for a (possibly multi-token) number.

    Positional weighting: for tokens ["5", "46"] the coordinate is
        E["5"] * 100 + E["46"] * 1
    where E[t] = Σ_{k: int token} softmax(logits[pos])[k] * value(k).
    This mirrors standard positional notation and is fully differentiable.
    """
    suffix_len = sum(len(s) for s in tok_strs)  # total digit count
    result = logits.new_zeros(())
    for s, pos in zip(tok_strs, positions.tolist()):
        suffix_len -= len(s)
        weight = 10.0 ** suffix_len
        subset_logits = logits[pos][int_ids]
        probs = F.softmax(subset_logits, dim=-1)
        ev = (probs * int_vals).sum()
        result = result + weight * ev
    return result


def extract_expected_click_seqs(
    logits: torch.Tensor,    # (seq_len, vocab_size)
    input_ids: torch.Tensor, # (seq_len,)
    labels: torch.Tensor,    # (seq_len,)
    target_text: str,
    tokenizer,
    int_ids: torch.Tensor,   # on CPU; moved to device inside
    int_vals: torch.Tensor,  # on CPU; moved to device inside
) -> tuple[Optional[torch.Tensor], list[tuple[float, float]]]:
    """
    Extract differentiable expected (x, y) tensors and ground-truth pairs.

    Only assistant-response token positions (labels != -100) are searched, so
    the function cannot accidentally match user-prompt coordinates.

    Returns:
        pred_coords : (N, 2) float tensor (differentiable) or None
        gt_coords   : list of (x, y) ground-truth floats
    """
    clicks = _RE_CLICK.findall(target_text)
    if not clicks:
        return None, []

    # Restrict search to the assistant response region
    asst_pos = (labels != -100).nonzero(as_tuple=True)[0]  # (L,)
    if len(asst_pos) == 0:
        return None, []
    asst_ids = input_ids[asst_pos]  # (L,) — assistant token IDs only

    int_ids_d = int_ids.to(logits.device)
    int_vals_d = int_vals.to(logits.device, dtype=logits.dtype)

    pred_list: list[torch.Tensor] = []
    gt_list: list[tuple[float, float]] = []
    search_off = 0

    for x_str, y_str in clicks:
        # Tokenise each coordinate string (no special tokens)
        x_toks = tokenizer.encode(x_str, add_special_tokens=False)
        y_toks = tokenizer.encode(y_str, add_special_tokens=False)

        # Find x span in the assistant region
        x_start = _find_subseq(asst_ids, x_toks, search_off)
        if x_start is None:
            continue
        x_seq_pos = asst_pos[x_start : x_start + len(x_toks)]
        x_tok_strs = [tokenizer.decode([t]).strip() for t in x_toks]

        # Find y span after x (skip at least the comma token)
        y_start = _find_subseq(asst_ids, y_toks, x_start + len(x_toks) + 1)
        if y_start is None:
            continue
        y_seq_pos = asst_pos[y_start : y_start + len(y_toks)]
        y_tok_strs = [tokenizer.decode([t]).strip() for t in y_toks]

        pred_x = _expected_coord(logits, x_seq_pos, x_tok_strs, int_ids_d, int_vals_d)
        pred_y = _expected_coord(logits, y_seq_pos, y_tok_strs, int_ids_d, int_vals_d)
        pred_list.append(torch.stack([pred_x, pred_y]))
        gt_list.append((float(x_str), float(y_str)))

        search_off = y_start + len(y_toks)

    if not pred_list:
        return None, []
    return torch.stack(pred_list), gt_list  # (N, 2), [(x,y),...]


def compute_batch_sdtw_loss(
    logits: torch.Tensor,       # (B, seq_len, vocab_size)
    input_ids: torch.Tensor,    # (B, seq_len)
    labels: torch.Tensor,       # (B, seq_len)
    target_texts: list[str],
    tokenizer,
    int_ids: torch.Tensor,
    int_vals: torch.Tensor,
    gamma: float = 1.0,
    coord_scale: float = 1000.0,
) -> Optional[torch.Tensor]:
    """
    Mean Soft-DTW loss over samples in the batch that contain click sequences.
    Coordinates are divided by coord_scale so distances are O(1).
    Returns None when no click sequences exist in the batch.
    """
    losses: list[torch.Tensor] = []
    for i, tgt in enumerate(target_texts):
        pred_coords, gt_pairs = extract_expected_click_seqs(
            logits[i], input_ids[i], labels[i], tgt, tokenizer, int_ids, int_vals
        )
        if pred_coords is None or len(gt_pairs) == 0:
            continue
        gt_t = torch.tensor(
            gt_pairs, dtype=pred_coords.dtype, device=pred_coords.device
        )
        losses.append(
            soft_dtw_loss(pred_coords / coord_scale, gt_t / coord_scale, gamma=gamma)
        )
    if not losses:
        return None
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & batch builder
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_set_max_pixels(processor: AutoProcessor, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
        print(f"Set image_processor.max_pixels = {max_pixels}")
    else:
        print("Warning: could not set max_pixels on this processor; ignoring --max_pixels")


_SEP = " | <sep> |"


class VCoTJsonDataset(Dataset):
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
            # Strip " | <sep> | ANSWER" — keep only the click sequence
            sep_pos = target.find(_SEP)
            if sep_pos != -1:
                target = target[:sep_pos].strip()
        return {"image": img, "prompt": r["prompt"], "target": target}


def build_batch(
    samples: list[dict], processor: AutoProcessor, device: torch.device
) -> dict:
    """
    Tokenise user+assistant messages; mask user/padding tokens with −100 in labels
    so cross-entropy is computed only on assistant tokens.
    """
    images = [s["image"] for s in samples]
    prompt_texts: list[str] = []
    full_texts: list[str] = []

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

    for i, (prompt_t, img) in enumerate(zip(prompt_texts, images)):
        pt = processor(
            text=[prompt_t], images=[img], return_tensors="pt", padding=False
        )
        plen = pt["input_ids"].shape[1]
        
        ids = input_ids[i]
        non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
        if len(non_pad) == 0:
            continue
        start = int(non_pad[0])
        pr = pt["input_ids"][0]
        del pt  # free pixel_values immediately; only plen is needed
        j = 0
        while j < plen and start + j < len(ids) and ids[start + j] == pr[j]:
            j += 1
        labels[i, : start + j] = -100

    if pad_id is not None:
        labels[labels == pad_id] = -100

    batch["labels"] = labels
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Data splitting & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_json_rows(path: str) -> list[dict]:
    with open(_REPO / path if not os.path.isabs(path) else path) as f:
        return json.load(f)


def train_val_test_split(
    rows: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be ≥ 0 and sum to < 1")
    n = len(rows)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Split leaves no training rows; reduce ratios or use more data")
    train_i = idx[n_test + n_val :]
    val_i = idx[n_test : n_test + n_val]
    test_i = idx[:n_test]
    return [rows[i] for i in train_i], [rows[i] for i in val_i], [rows[i] for i in test_i]


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    processor: AutoProcessor,
    device: torch.device,
    int_ids: Optional[torch.Tensor] = None,
    int_vals: Optional[torch.Tensor] = None,
    sdtw_gamma: float = 1.0,
    sdtw_lambda: float = 0.0,
) -> dict[str, float | None]:
    """Return mean CE loss and (optionally) mean Soft-DTW over the loader."""
    if len(loader) == 0:
        return {"ce_loss": None, "sdtw_loss": None}
    model.eval()
    ce_total, sdtw_total, n_sdtw, n = 0.0, 0.0, 0, 0
    use_sdtw = sdtw_lambda > 0 and int_ids is not None

    for raw in loader:
        batch = build_batch(raw, processor, device)
        out = model(**batch)
        ce_total += float(out.loss.item())
        n += 1
        if use_sdtw:
            target_texts = [s["target"] for s in raw]
            sdtw_val = compute_batch_sdtw_loss(
                out.logits, batch["input_ids"], batch["labels"],
                target_texts, processor.tokenizer, int_ids, int_vals,
                gamma=sdtw_gamma,
            )
            if sdtw_val is not None:
                sdtw_total += float(sdtw_val.item())
                n_sdtw += 1

    model.train()
    return {
        "ce_loss": ce_total / n,
        "sdtw_loss": sdtw_total / n_sdtw if n_sdtw > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data / checkpointing
    p.add_argument("--dataset", default="data/vcot_dataset_unique.json")
    p.add_argument("--output_dir", default="runs/qwen_lora")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Use only first N rows (debug).")
    p.add_argument(
        "--clicks_only",
        action=argparse.BooleanOptionalAction, default=True,
        help="Strip the '| <sep> | ANSWER' suffix from targets so the model only "
             "predicts the click sequence. Use --no-clicks_only to keep the answer.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Micro-batch size; 1 recommended on laptop/MPS.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Effective batch ≈ batch_size × N.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction, default=True,
        help="Trade compute for memory during backward (recommended).",
    )
    p.add_argument(
        "--max_pixels", type=int, default=401408,
        help="Cap image_processor.max_pixels. Default 401408 ≈ 512×28². "
             "Reduce further (e.g. 200704) if you still hit RAM limits.",
    )

    # LoRA / QLoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--qlora", action="store_true", default=False,
        help="Enable 4-bit NF4 quantisation (QLoRA). CUDA only; falls back to fp16 on MPS/CPU.",
    )
    
    p.add_argument(
        "--lora_vision_projector", action="store_true", default=False,
        help="Also apply LoRA to the Qwen2-VL visual-merger MLP layers.",
    )
    p.add_argument(
        "--freeze_vision",
        action=argparse.BooleanOptionalAction, default=True,
        help="Freeze the ViT backbone (visual.* except visual.merger). Saves memory.",
    )

    # Soft-DTW
    p.add_argument(
        "--sdtw_lambda", type=float, default=0.1,
        help="Weight λ for the Soft-DTW auxiliary loss (0 = CE only).",
    )
    p.add_argument(
        "--sdtw_gamma", type=float, default=0.1,
        help="Soft-DTW smoothness γ. Larger → softer path selection. "
             "Smaller values keep the loss non-negative and closer to true DTW.",
    )
    p.add_argument(
        "--coord_scale", type=float, default=1000.0,
        help="Divide click coordinates by this before computing Soft-DTW distances.",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device & dtype ────────────────────────────────────────────────────────
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    # bfloat16 on CUDA (QLoRA compute dtype); float16 on MPS; float32 on CPU
    if device.type == "cuda":
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Data ──────────────────────────────────────────────────────────────────
    rows = load_json_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    train_rows, val_rows, test_rows = train_val_test_split(
        rows, args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"Split: train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}"
        f"  (seed={args.seed})"
    )

    train_loader = DataLoader(
        VCoTJsonDataset(train_rows, clicks_only=args.clicks_only), batch_size=args.batch_size,
        shuffle=True, collate_fn=lambda b: b,
    )
    val_loader = DataLoader(
        VCoTJsonDataset(val_rows, clicks_only=args.clicks_only), batch_size=args.batch_size,
        shuffle=False, collate_fn=lambda b: b,
    )

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.model_name)
    _maybe_set_max_pixels(processor, args.max_pixels)

    # Build integer-token map for Soft-DTW expected coordinates
    int_ids, int_vals = build_int_token_map(processor.tokenizer)
    print(f"Integer-valued vocabulary tokens found: {len(int_ids)}")

    # ── Model — QLoRA (CUDA) or fp16 LoRA (MPS/CPU) ───────────────────────────
    if args.qlora and device.type == "cuda":
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        print("Loading model in 4-bit NF4 (QLoRA)…")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        if args.qlora:
            print(
                "Warning: --qlora requires CUDA; "
                f"running on {device.type} with fp{dtype} LoRA instead."
            )
        print(f"Loading model in {dtype}…")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype
        ).to(device)

    # ── Freeze ViT backbone ───────────────────────────────────────────────────
    # Must happen BEFORE get_peft_model so frozen params don't get LoRA adapters.
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            # Freeze all visual.* layers except the visual.merger projector
            if "visual" in name and "merger" not in name:
                param.requires_grad_(False)
                frozen += 1
        print(f"Frozen {frozen} ViT backbone parameter tensors (visual.* except merger).")

    # ── LoRA config ───────────────────────────────────────────────────────────
    target_modules = [
        # LM self-attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # LM feed-forward
        "gate_proj", "up_proj", "down_proj",
    ]
    if args.lora_vision_projector:
        # Qwen2-VL visual merger MLP: nn.Sequential(Linear, GELU, Linear)
        # The two Linear layers sit at index 0 and 2 within merger.mlp.
        # PEFT matches by name suffix, so "mlp.0" / "mlp.2" is sufficient because
        # the LM uses named projections (gate_proj etc.), not indexed MLPs.
        target_modules += ["mlp.0", "mlp.2"]
        print("LoRA also targeting visual.merger MLP (mlp.0, mlp.2).")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing (skip if already set by prepare_model_for_kbit_training)
    if args.gradient_checkpointing and not (args.qlora and device.type == "cuda"):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # ── Output directories & manifests ────────────────────────────────────────
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "split_manifest.json", "w") as f:
        json.dump(
            {
                "dataset": args.dataset, "seed": args.seed,
                "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
                "train_n": len(train_rows), "val_n": len(val_rows), "test_n": len(test_rows),
            },
            f, indent=2,
        )

    test_path = out_root / "test_holdout.json"
    with open(test_path, "w") as f:
        json.dump(test_rows, f)
    print(f"Held-out test: {test_path} ({len(test_rows)} samples)")

    # Save training config for reproducibility
    with open(out_root / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_dir = out_root / "best"
    last_dir = out_root / "last"
    best_val: float | None = None
    use_sdtw = args.sdtw_lambda > 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    accum = max(1, args.gradient_accumulation_steps)

    for epoch in range(args.epochs):
        ce_total = sdtw_total = n_sdtw = 0.0
        n_steps = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for step, raw in enumerate(pbar):
            batch = build_batch(raw, processor, device)
            out = model(**batch)
            ce_loss = out.loss  # CE on assistant tokens only

            # ── Soft-DTW auxiliary loss ──────────────────────────────────────
            sdtw_val: Optional[torch.Tensor] = None
            if use_sdtw:
                target_texts = [s["target"] for s in raw]
                sdtw_val = compute_batch_sdtw_loss(
                    out.logits, batch["input_ids"], batch["labels"],
                    target_texts, processor.tokenizer, int_ids, int_vals,
                    gamma=args.sdtw_gamma, coord_scale=args.coord_scale,
                )

            # Free the large logits tensor before backward — it is no longer needed
            del out

            total_loss = ce_loss
            if sdtw_val is not None:
                total_loss = ce_loss + args.sdtw_lambda * sdtw_val

            (total_loss / accum).backward()

            ce_scalar = float(ce_loss.item())
            sdtw_scalar = float(sdtw_val.item()) if sdtw_val is not None else None

            # Free all forward-pass tensors before the next step
            del batch, total_loss, ce_loss, sdtw_val

            ce_total += ce_scalar
            if sdtw_scalar is not None:
                sdtw_total += sdtw_scalar
                n_sdtw += 1
            n_steps += 1

            # Gradient accumulation step
            stepped = (step + 1) % accum == 0 or step == len(train_loader) - 1
            if stepped:
                opt.step()
                opt.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            postfix = {"ce": f"{ce_scalar:.4f}"}
            if sdtw_scalar is not None:
                postfix["sdtw"] = f"{sdtw_scalar:.4f}"
            pbar.set_postfix(**postfix)

        # ── Epoch summary ─────────────────────────────────────────────────────
        train_ce = ce_total / max(n_steps, 1)
        train_sdtw = sdtw_total / n_sdtw if n_sdtw > 0 else None

        val_metrics = evaluate_loader(
            model, val_loader, processor, device,
            int_ids=int_ids if use_sdtw else None,
            int_vals=int_vals if use_sdtw else None,
            sdtw_gamma=args.sdtw_gamma,
            sdtw_lambda=args.sdtw_lambda,
        )
        val_ce = val_metrics["ce_loss"]
        val_sdtw = val_metrics["sdtw_loss"]

        # Combined val metric for checkpoint selection
        if val_ce is not None and val_sdtw is not None:
            val_combined = val_ce + args.sdtw_lambda * val_sdtw
        elif val_ce is not None:
            val_combined = val_ce
        else:
            val_combined = None

        # Log
        sdtw_str = ""
        if train_sdtw is not None:
            sdtw_str += f"  train_sdtw={train_sdtw:.4f}"
        if val_sdtw is not None:
            sdtw_str += f"  val_sdtw={val_sdtw:.4f}"
        val_ce_str = f"{val_ce:.4f}" if val_ce is not None else "—"
        print(
            f"epoch {epoch + 1}  train_ce={train_ce:.4f}  val_ce={val_ce_str}{sdtw_str}"
        )

        # Checkpoint best model
        if val_combined is not None and not math.isnan(val_combined):
            if best_val is None or val_combined < best_val:
                best_val = val_combined
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                with open(best_dir / "best_metrics.json", "w") as f:
                    json.dump(
                        {
                            "epoch": epoch + 1,
                            "val_ce": val_ce,
                            "val_sdtw": val_sdtw,
                            "val_combined": val_combined,
                            "train_ce": train_ce,
                            "train_sdtw": train_sdtw,
                        },
                        f, indent=2,
                    )
                print(f"  → saved best checkpoint (val_combined={val_combined:.4f})")

    # ── Save last epoch ───────────────────────────────────────────────────────
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    print(f"Saved last epoch → {last_dir}")
    if best_val is not None:
        print(f"Best combined val loss = {best_val:.4f}  → {best_dir}")
    else:
        print("No validation loss tracked; use last/ or expand val set.")


if __name__ == "__main__":
    main()
