from __future__ import annotations

import argparse
import json
import math
import os
import re
import random
import warnings
from pathlib import Path
from typing import NamedTuple, Optional

import torch
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

    return dp[n][m] / (n + m)


def parse_clicks(text: str) -> list[tuple[float, float]]:
    """Extract (x, y) pairs from a click-sequence string."""
    return [(float(x), float(y)) for x, y in _RE_CLICK.findall(text)]


def soft_dtw(
    pred: torch.Tensor,    # (N, 2)
    target: torch.Tensor,  # (M, 2)
    gamma: float = 1.0,
    step_weights: Optional[torch.Tensor] = None,  # (M,)
) -> torch.Tensor:
    """Soft-DTW with optional per-target-step weighting.

    When *step_weights* is provided it scales each column of the cost
    matrix so that mismatching early GT steps (high weight) is more
    costly than mismatching late ones.
    """
    n, m = pred.shape[0], target.shape[0]
    if n == 0 or m == 0:
        return pred.sum() * 0.0

    diff = pred.unsqueeze(1) - target.unsqueeze(0)   # (N, M, 2)
    cost = diff.pow(2).sum(-1).clamp(min=1e-12).sqrt()  # (N, M)

    if step_weights is not None:
        cost = cost * step_weights.unsqueeze(0)  # broadcast (1, M)

    INF = torch.tensor(1e9, dtype=pred.dtype, device=pred.device)
    ZERO = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

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

    close_ids = tokenizer.encode("</click>", add_special_tokens=False)

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
        search_off = y_start + len(y_toks) + len(close_ids)
 
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset coordinate statistics for independent x/y normalisation
# ─────────────────────────────────────────────────────────────────────────────

_SEP = " | <sep> |"


class CoordStats(NamedTuple):
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


def compute_coord_stats(rows: list[dict], clicks_only: bool = True) -> CoordStats:
    """Compute position-level mean/std of x and y across all clicks in *rows*.

    Used to normalise x and y independently before computing spatial losses,
    so that the tighter y-distribution (σ_y ≈ 0.6 × σ_x in typical chart data)
    does not silently dominate the loss.
    """
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
    stats = CoordStats(
        x_mean=float(xs_t.mean()),
        x_std=max(float(xs_t.std()), 1.0),
        y_mean=float(ys_t.mean()),
        y_std=max(float(ys_t.std()), 1.0),
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary losses for breaking mode collapse
# ─────────────────────────────────────────────────────────────────────────────

def repetition_penalty_loss(
    pred_seq: torch.Tensor,
    min_dist: float = 0.1,
) -> torch.Tensor:
    """Soft hinge penalty on consecutive predicted points that are closer than
    *min_dist* (in normalised coordinate space).

    Directly combats the dominant failure mode where the model predicts the
    same (x, y) at every step, collapsing to the dataset mean.
    """
    if pred_seq.shape[0] < 2:
        return pred_seq.new_zeros(())
    diffs = pred_seq[1:] - pred_seq[:-1]
    dists = diffs.pow(2).sum(-1).clamp(min=1e-12).sqrt()
    return F.relu(min_dist - dists).mean()


def step_position_weights(
    n_steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    decay: float = 0.9,
) -> torch.Tensor:
    """Exponentially-decaying per-step weights (normalised to mean = 1).

    Early steps carry higher weight because they capture the exploratory
    jumps (median ≈ 64 px) that are critical for chart comprehension,
    while late steps are local fixations (median ≈ 40 px by step 30).
    """
    w = torch.pow(
        torch.tensor(decay, dtype=dtype, device=device),
        torch.arange(n_steps, dtype=dtype, device=device),
    )
    return w / w.mean()


def coverage_loss(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
) -> torch.Tensor:

    if pred_seq.shape[0] < 2:
        return pred_seq.new_zeros(())
    pred_var = pred_seq.var(dim=0) 
    gt_var = gt_seq.var(dim=0)
    return F.relu(gt_var - pred_var).sum()


def velocity_matching_loss(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Soft-DTW applied to the *deltas* (velocities) of both sequences.

    Standard position-DTW can score two sequences identically even if one
    is static (repeated coords) and the other moves.  This term directly
    penalises zero-delta predictions when the GT has non-zero movement,
    breaking the "predict the mean at every step" failure mode.
    """
    if pred_seq.shape[0] < 2 or gt_seq.shape[0] < 2:
        return pred_seq.new_zeros(())
    pred_v = pred_seq[1:] - pred_seq[:-1]
    gt_v   = gt_seq[1:]   - gt_seq[:-1]
    return soft_dtw(pred_v, gt_v, gamma=gamma)


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level sequence loss (all components)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_coords(
    coords: torch.Tensor,
    stats: Optional[CoordStats],
    fallback_scale: float,
) -> torch.Tensor:
    """Divide x by σ_x and y by σ_y for equal contribution, or fall back to
    a single scalar divisor when stats are unavailable."""
    if stats is not None:
        scale = torch.tensor(
            [stats.x_std, stats.y_std],
            device=coords.device, dtype=coords.dtype,
        )
        return coords / scale
    return coords / fallback_scale


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
    coord_stats: Optional[CoordStats] = None,
    step_weight_decay: float = 0.0,
    rep_min_dist: float = 0.1,
    compute_repetition: bool = False,
    compute_coverage: bool = False,
    compute_velocity: bool = False,
) -> dict[str, Optional[torch.Tensor]]:
    """Return a dict of loss components: coord, repetition, coverage, velocity.

    Each key maps to the mean loss across the batch for that component,
    or None if no valid samples contributed.
    """
    coord_losses: list[torch.Tensor] = []
    rep_losses:   list[torch.Tensor] = []
    cov_losses:   list[torch.Tensor] = []
    vel_losses:   list[torch.Tensor] = []

    for i, tgt in enumerate(target_texts):

        pred_coords, gt_pairs = extract_expected_click_seqs(
            logits[i], input_ids[i], labels[i], tgt, tokenizer, int_ids, int_vals,
            max_clicks=max_coord_clicks,
        )
    
        if pred_coords is None or not gt_pairs:
            continue

        gt_t = torch.tensor(
            gt_pairs, dtype=pred_coords.dtype, device=pred_coords.device
        )

        pred_s = _normalise_coords(pred_coords, coord_stats, coord_scale)
        gt_s   = _normalise_coords(gt_t,        coord_stats, coord_scale)

        # Step weights (emphasise early exploratory clicks)
        sw: Optional[torch.Tensor] = None
        if step_weight_decay > 0:
            sw = step_position_weights(
                len(gt_s), pred_s.device, pred_s.dtype, step_weight_decay
            )

        # ── Main coordinate loss ──
        if use_sdtw:
            coord_losses.append(soft_dtw(pred_s, gt_s, gamma=sdtw_gamma,
                                         step_weights=sw))
        else:
            n = min(len(pred_s), len(gt_s))
            diff = pred_s[:n] - gt_s[:n]
            per_step = diff.pow(2).sum(-1).add(1e-12).sqrt()
            if sw is not None:
                per_step = per_step * sw[:n]
            coord_losses.append(per_step.mean())

        # ── Repetition penalty ──
        print('pred_s',pred_s)
        print('gt_s',gt_s)
        hh
        if compute_repetition:
            rep_losses.append(
                repetition_penalty_loss(pred_s, min_dist=rep_min_dist)
            )

        # ── Coverage ──
        if compute_coverage:
            cov_losses.append(coverage_loss(pred_s, gt_s))

        # ── Velocity matching ──
        if compute_velocity:
            vel_losses.append(
                velocity_matching_loss(pred_s, gt_s, gamma=sdtw_gamma)
            )

    return {
        "coord":      torch.stack(coord_losses).mean() if coord_losses else None,
        "repetition": torch.stack(rep_losses).mean()    if rep_losses   else None,
        "coverage":   torch.stack(cov_losses).mean()    if cov_losses   else None,
        "velocity":   torch.stack(vel_losses).mean()    if vel_losses   else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

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


def _gen_eval_slice(rows: list[dict], n: int, start: int) -> list[dict]:
    if not rows or n <= 0:
        return []
    L = len(rows)
    start = start % L
    return [rows[(start + i) % L] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Inference-time repetition blocker
# ─────────────────────────────────────────────────────────────────────────────

class ClickRepetitionBlocker(LogitsProcessor):
    """Penalise digit tokens that would exactly reproduce the coordinates of the
    most recently completed <click> during autoregressive generation.

    Fires only when the model is currently inside an open <click>…</click> span
    and there is at least one preceding completed click to compare against.
    The penalty is applied to the exact token-ID(s) that encode the repeated
    coordinate string, making the model fall back to its next-best prediction.
    """

    _RE = re.compile(r"<click>(\d+),(\d+)</click>")

    def __init__(
        self,
        prompt_length: int,
        tokenizer,
        penalty: float = 50.0,
    ) -> None:
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.penalty = penalty

    def __call__(
        self,
        input_ids: torch.LongTensor,   # (batch, seq_so_far)
        scores: torch.FloatTensor,     # (batch, vocab_size)
    ) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            gen_ids = input_ids[b, self.prompt_length :]
            gen_text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)

            # Collect all completed clicks generated so far
            clicks = self._RE.findall(gen_text)
            if not clicks:
                continue  # nothing to block yet

            # Are we currently inside an open (incomplete) <click> tag?
            open_pos  = gen_text.rfind("<click>")
            close_pos = gen_text.rfind("</click>")
            if open_pos == -1 or open_pos <= close_pos:
                continue  # not inside a new click

            prev_x, prev_y = clicks[-1]
            # What has been emitted inside the open <click> so far?
            partial = gen_text[open_pos + len("<click>") :]
            # Block the previous x while generating x; block the previous y
            # once we've passed the comma separator.
            blocked_str = prev_x if "," not in partial else prev_y
            blocked_ids = self.tokenizer.encode(blocked_str, add_special_tokens=False)

            for tid in set(blocked_ids):
                if 0 <= tid < scores.shape[-1]:
                    scores[b, tid] = scores[b, tid] - self.penalty

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

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
            # Qwen2-VL hub generation_config uses top_k=1; without override, sampling is one-token (greedy).
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

        img_w = s.get("img_w", 0)
        img_h = s.get("img_h", 0)
        img_label = s["image"] if isinstance(s["image"], str) else f"PIL({img_w}x{img_h})"

        per_sample.append({
            "idx":         i,
            "image":       img_label,
            "prompt":      s["prompt"],
            "gt_clicks":   gt_clicks,
            "pred_clicks": pred_clicks,
            "dtw":         score,
            "len_err":     len_err,
        })

        if verbose:
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
    max_seq_len: Optional[int] = None,
    coord_stats: Optional[CoordStats] = None,
    step_weight_decay: float = 0.0,
    rep_min_dist: float = 0.1,
    rep_lambda: float = 0.0,
    coverage_lambda: float = 0.0,
    velocity_lambda: float = 0.0,
) -> dict[str, float | None]:
    if len(loader) == 0:
        return {"ce_loss": None, "coord_loss": None,
                "rep_loss": None, "cov_loss": None, "vel_loss": None}

    model.eval()
    ce_total = coord_total = rep_total = cov_total = vel_total = 0.0
    n = n_coord = n_rep = n_cov = n_vel = skipped = 0
    use_coord = coord_lambda > 0 and int_ids is not None

    for raw in loader:
        batch = build_batch(raw, processor, device, system_prompt=system_prompt)
        if max_seq_len is not None and batch["input_ids"].shape[1] > max_seq_len:
            skipped += 1
            del batch
            continue
        out = model(**batch)
        ce_total += float(out.loss.item())
        n += 1

        if use_coord:

            target_texts = [s["target"] for s in raw]
            loss_dict = compute_batch_sequence_loss(
                out.logits, batch["input_ids"], batch["labels"],
                target_texts, processor.tokenizer, int_ids, int_vals,
                coord_scale=coord_scale, use_sdtw=use_sdtw,
                sdtw_gamma=sdtw_gamma,
                max_coord_clicks=max_coord_clicks,
                coord_stats=coord_stats,
                step_weight_decay=step_weight_decay,
                rep_min_dist=rep_min_dist,
                compute_repetition=rep_lambda > 0,
                compute_coverage=coverage_lambda > 0,
                compute_velocity=velocity_lambda > 0,
            )
            if loss_dict["coord"] is not None:
                coord_total += float(loss_dict["coord"].item())
                n_coord += 1
            if loss_dict["repetition"] is not None:
                rep_total += float(loss_dict["repetition"].item())
                n_rep += 1
            if loss_dict["coverage"] is not None:
                cov_total += float(loss_dict["coverage"].item())
                n_cov += 1
            if loss_dict["velocity"] is not None:
                vel_total += float(loss_dict["velocity"].item())
                n_vel += 1

    model.train()
    if skipped:
        print(f"  [evaluate_loader] skipped {skipped} batches (seq_len > {max_seq_len})")
    return {
        "ce_loss":    ce_total   / n       if n       > 0 else None,
        "coord_loss": coord_total / n_coord if n_coord > 0 else None,
        "rep_loss":   rep_total   / n_rep   if n_rep   > 0 else None,
        "cov_loss":   cov_total   / n_cov   if n_cov   > 0 else None,
        "vel_loss":   vel_total   / n_vel   if n_vel   > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler / IO / Split helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data / checkpointing
    p.add_argument("--dataset",    default="data/vcot_dataset_unique.json")
    p.add_argument("--output_dir", default="runs/qwen_lora")
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a saved LoRA adapter directory (e.g. runs/qwen_lora_v3/best). "
             "Loads adapter weights; use with --epochs and --first_epoch to continue training.",
    )
    p.add_argument(
        "--first_epoch",
        type=int,
        default=1,
        help="1-based index of the first epoch in this run (e.g. 2 after finishing epoch 1).",
    )
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
    p.add_argument(
        "--max_seq_len", type=int, default=None,
        help="Skip training (and val) batches whose tokenized length exceeds this. "
             "Useful for skipping slow large-image samples. None = no limit.",
    )

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

    # ── Spatial coordinate loss ──────────────────────────────────────────────
    p.add_argument("--coord_lambda", type=float, default=0.1,
                   help="Weight λ for spatial auxiliary loss (0 = CE only).")
    p.add_argument("--coord_scale",  type=float, default=1000.0,
                   help="Divide coordinates by this before computing distances "
                        "(used only when --no-normalize_xy).")
    p.add_argument("--max_coord_clicks", type=int, default=15,
                   help="Max clicks per sample for coord loss (0 = no limit).")
    p.add_argument(
        "--use_sdtw", action=argparse.BooleanOptionalAction, default=True,
        help="Use soft-DTW instead of pointwise Euclidean.",
    )
    p.add_argument("--sdtw_gamma", type=float, default=1.0,
                   help="Smoothing factor for soft-DTW.")

    # ── Independent x/y normalisation (#4) ───────────────────────────────────
    p.add_argument(
        "--normalize_xy",
        action=argparse.BooleanOptionalAction, default=True,
        help="Normalise x and y independently by their dataset std before "
             "computing spatial losses.  Prevents the tighter y-distribution "
             "from dominating (σ_y ≈ 0.6 × σ_x in typical chart data).",
    )

    # ── Repetition penalty (#1) ──────────────────────────────────────────────
    p.add_argument("--rep_lambda", type=float, default=0.3,
                   help="Weight for repetition penalty loss. 0 = disabled.")
    p.add_argument("--rep_min_dist", type=float, default=0.15,
                   help="Min distance between consecutive predictions in "
                        "normalised space.  Below this the penalty kicks in.")

    # ── Step-position weighting (#2) ─────────────────────────────────────────
    p.add_argument("--step_weight_decay", type=float, default=0.9,
                   help="Exponential decay applied per GT step so early "
                        "exploratory jumps carry higher loss weight. "
                        "0 = disabled (uniform weighting).")

    # ── Coverage loss (#3) ───────────────────────────────────────────────────
    p.add_argument("--coverage_lambda", type=float, default=0.05,
                   help="Weight for coverage loss that penalises spatial "
                        "collapse below GT variance. 0 = disabled.")

    # ── Velocity matching (#5) ───────────────────────────────────────────────
    p.add_argument("--velocity_lambda", type=float, default=0.05,
                   help="Weight for soft-DTW on deltas (velocities). "
                        "Penalises zero-movement predictions. 0 = disabled.")

    # ── Scheduled sampling (#3) ───────────────────────────────────────────────
    p.add_argument(
        "--ss_prob", type=float, default=0.0,
        help="Scheduled-sampling probability: fraction of teacher-forced assistant "
             "tokens replaced by the model's own greedy prediction each step. "
             "0 = pure teacher forcing (disabled).  Typical useful range: 0.1–0.3.",
    )
    p.add_argument(
        "--ss_anneal_epochs", type=int, default=0,
        help="If > 0, linearly ramp --ss_prob from 0 to its target value over "
             "this many epochs, then hold constant.  0 = use ss_prob from epoch 1.",
    )

    # ── Inference repetition blocking (#4) ───────────────────────────────────
    p.add_argument(
        "--gen_block_repeats",
        action=argparse.BooleanOptionalAction, default=True,
        help="At inference, apply ClickRepetitionBlocker: suppress digit tokens "
             "that would exactly reproduce the previous click's coordinate.",
    )
    p.add_argument(
        "--gen_block_repeat_penalty", type=float, default=50.0,
        help="Logit penalty applied by ClickRepetitionBlocker to repeated "
             "coordinate tokens.  Higher = stronger blocking.",
    )

    # Generation eval
    p.add_argument("--gen_eval_epochs", type=int, default=1,
                   help="Run generation-based (DTW) eval every N epochs. 0 = never.")
    p.add_argument("--gen_eval_steps", type=int, default=0,
                   help="Also run generation eval every N optimizer steps during training. 0 = never.")
    p.add_argument("--gen_eval_samples", type=int, default=16,
                   help="Number of val samples to use for generation eval.")
    p.add_argument("--gen_max_new_tokens", type=int, default=512)
    p.add_argument(
        "--gen_temperature", type=float, default=0.3,
        help="Generation temperature. 0 = greedy (deterministic). >0 enables sampling.",
    )
    p.add_argument(
        "--gen_repetition_penalty", type=float, default=1.0,
        help="Repetition penalty during generation. >1.0 discourages repeated coordinates.",
    )
    p.add_argument(
        "--gen_eval_fixed_slice", action="store_true",
        help="If set, always use the first --gen_eval_samples val rows (no rotation).",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Scheduled sampling
# ─────────────────────────────────────────────────────────────────────────────

def scheduled_sampling_forward(
    model,
    batch: dict,
    ss_prob: float,
) -> object:

    if ss_prob <= 0.0:
        return model(**batch)

    # Pass 1 — collect greedy predictions without computing gradients
    with torch.no_grad():
        out1 = model(**batch)
        greedy_ids = out1.logits.argmax(dim=-1)  # (B, T); logits[t] → token t+1
    del out1

    # Replace teacher-forced tokens in the assistant region
    new_input_ids = batch["input_ids"].clone()
    labels = batch["labels"]
    B, T = new_input_ids.shape
    for b in range(B):
        for t in range(T - 1):
            # labels[b, t+1] != -100 iff t+1 is an assistant (supervised) token
            if labels[b, t + 1] != -100 and random.random() < ss_prob:
                new_input_ids[b, t + 1] = greedy_ids[b, t]

    # Pass 2 — differentiable forward on the corrupted sequence
    corrupted = {**batch, "input_ids": new_input_ids}
    return model(**corrupted)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.first_epoch < 1:
        raise ValueError("--first_epoch must be >= 1")

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
    print(
        f"Split → train={len(train_rows)}  val={len(val_rows)}  "
        f"test={len(test_rows)}  (seed={args.seed})"
    )

    # ── Coordinate statistics for independent normalisation ───────────────────
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=lambda b: b
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b
    )

    # ── Processor ─────────────────────────────────────────────────────────────
    if resume_path is not None:
        print(f"Loading processor from resumed checkpoint: {resume_path}")
        processor = Qwen2VLProcessor.from_pretrained(str(resume_path), use_fast=False)
    else:
        processor = Qwen2VLProcessor.from_pretrained(args.model_name, use_fast=False)
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

    effective_dropout = 0.0 if device.type == "mps" else args.lora_dropout
    if device.type == "mps" and args.lora_dropout > 0.0:
        print(f"MPS detected: overriding lora_dropout {args.lora_dropout} → 0.0 "
              "(MPS dropout backward is unreliable).")

    if resume_path is not None:
        print(f"Resuming LoRA weights from {resume_path}")
        # Default is_trainable=False freezes all params → empty optimizer; must train adapters.
        model = PeftModel.from_pretrained(
            model, str(resume_path), is_trainable=True
        )
        model.print_trainable_parameters()
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

    train_cfg = vars(args).copy()
    if coord_stats is not None:
        train_cfg["coord_stats"] = coord_stats._asdict()
    with open(out_root / "train_config.json", "w") as f:
        json.dump(train_cfg, f, indent=2)

    print(f"Held-out test set: {out_root / 'test_holdout.json'} ({len(test_rows)} samples)")

    # ── Derived flags ─────────────────────────────────────────────────────────
    best_dir = out_root / "best"
    last_dir = out_root / "last"
    best_val: float | None = None
    if resume_path is not None:
        metrics_file = resume_path / "best_metrics.json"
        if metrics_file.is_file():
            try:
                with open(metrics_file) as f:
                    prev = json.load(f)
                vc = prev.get("val_combined")
                if vc is not None and isinstance(vc, (int, float)) and not math.isnan(vc):
                    best_val = float(vc)
                    print(f"Loaded previous best val_combined={best_val:.4f} from {metrics_file}")
            except (json.JSONDecodeError, OSError, TypeError):
                pass
    use_coord = args.coord_lambda > 0.0
    use_rep   = args.rep_lambda > 0.0
    use_cov   = args.coverage_lambda > 0.0
    use_vel   = args.velocity_lambda > 0.0

    active_losses = ["CE"]
    if use_coord: active_losses.append(f"coord(λ={args.coord_lambda})")
    if use_rep:   active_losses.append(f"rep(λ={args.rep_lambda}, d={args.rep_min_dist})")
    if use_cov:   active_losses.append(f"cov(λ={args.coverage_lambda})")
    if use_vel:   active_losses.append(f"vel(λ={args.velocity_lambda})")
    if args.step_weight_decay > 0:
        active_losses.append(f"step_wt(decay={args.step_weight_decay})")
    if coord_stats is not None:
        active_losses.append("xy_norm")
    if args.ss_prob > 0:
        active_losses.append(
            f"sched_samp(p={args.ss_prob}"
            + (f",anneal={args.ss_anneal_epochs}ep" if args.ss_anneal_epochs > 0 else "")
            + ")"
        )
    print(f"Active losses: {' + '.join(active_losses)}")
    if args.gen_block_repeats:
        print(f"Inference: ClickRepetitionBlocker enabled (penalty={args.gen_block_repeat_penalty})")

    epoch_last = args.first_epoch + args.epochs - 1

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    grad_norm: float = 0.0
    skipped_steps: int = 0

    for epoch in range(args.epochs):
        epoch_display = args.first_epoch + epoch

        # Scheduled-sampling probability for this epoch (linear ramp if requested)
        if args.ss_anneal_epochs > 0:
            current_ss_prob = args.ss_prob * min(1.0, epoch / max(args.ss_anneal_epochs, 1))
        else:
            current_ss_prob = args.ss_prob
        if args.ss_prob > 0:
            print(f"  Scheduled sampling: ss_prob={current_ss_prob:.4f} "
                  f"(target={args.ss_prob}, epoch {epoch_display})")

        ce_total = coord_total = rep_total = cov_total = vel_total = 0.0
        n_steps = n_coord = n_rep = n_cov = n_vel = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch_display}/{epoch_last}")

        for step, raw in enumerate(pbar):
            batch = build_batch(raw, processor, device, system_prompt=args.system_prompt)
            seq_len = batch["input_ids"].shape[1]
            if args.max_seq_len is not None and seq_len > args.max_seq_len:
                skipped_steps += 1
                pbar.write(
                    f"  [skip] step {step}: seq_len={seq_len} > max_seq_len={args.max_seq_len}"
                )
                del batch
                if device.type == "mps":
                    torch.mps.empty_cache()
                continue
            out = scheduled_sampling_forward(model, batch, current_ss_prob)
            ce_loss = out.loss

            # ── Auxiliary spatial losses ───────────────────────────────────
            coord_val: Optional[torch.Tensor] = None
            rep_val:   Optional[torch.Tensor] = None
            cov_val:   Optional[torch.Tensor] = None
            vel_val:   Optional[torch.Tensor] = None

            if use_coord or use_rep or use_cov or use_vel:
                target_texts = [s["target"] for s in raw]
                loss_dict = compute_batch_sequence_loss(
                    out.logits, batch["input_ids"], batch["labels"],
                    target_texts, processor.tokenizer, int_ids, int_vals,
                    coord_scale=args.coord_scale,
                    use_sdtw=args.use_sdtw,
                    sdtw_gamma=args.sdtw_gamma,
                    max_coord_clicks=args.max_coord_clicks,
                    coord_stats=coord_stats,
                    step_weight_decay=args.step_weight_decay,
                    rep_min_dist=args.rep_min_dist,
                    compute_repetition=use_rep,
                    compute_coverage=use_cov,
                    compute_velocity=use_vel,
                )
                coord_val = loss_dict["coord"]
                rep_val   = loss_dict["repetition"]
                cov_val   = loss_dict["coverage"]
                vel_val   = loss_dict["velocity"]

            del out

            # ── Combine losses ────────────────────────────────────────────
            total_loss = ce_loss
            if coord_val is not None:
                total_loss = total_loss + args.coord_lambda * coord_val
            if rep_val is not None:
                total_loss = total_loss + args.rep_lambda * rep_val
            if cov_val is not None:
                total_loss = total_loss + args.coverage_lambda * cov_val
            if vel_val is not None:
                total_loss = total_loss + args.velocity_lambda * vel_val

            (total_loss / accum).backward()

            # ── Bookkeeping ───────────────────────────────────────────────
            ce_s    = float(ce_loss.item())
            coord_s = float(coord_val.item()) if coord_val is not None else None
            rep_s   = float(rep_val.item())   if rep_val   is not None else None
            cov_s   = float(cov_val.item())   if cov_val   is not None else None
            vel_s   = float(vel_val.item())   if vel_val   is not None else None
            del batch, total_loss, ce_loss, coord_val, rep_val, cov_val, vel_val

            ce_total += ce_s
            n_steps += 1
            if coord_s is not None:
                coord_total += coord_s; n_coord += 1
            if rep_s is not None:
                rep_total += rep_s; n_rep += 1
            if cov_s is not None:
                cov_total += cov_s; n_cov += 1
            if vel_s is not None:
                vel_total += vel_s; n_vel += 1

            # ── Gradient accumulation ─────────────────────────────────────
            is_last = step == len(train_loader) - 1
            if (step + 1) % accum == 0 or is_last:
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

            # ── Mid-training generation eval ──────────────────────────────
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
                        repetition_penalty=args.gen_repetition_penalty,
                        block_repeats=args.gen_block_repeats,
                        block_repeat_penalty=args.gen_block_repeat_penalty,
                    )

            # ── Progress bar ──────────────────────────────────────────────
            postfix: dict = {
                "ce": f"{ce_s:.4f}",
                "gnorm": f"{grad_norm:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if coord_s is not None: postfix["coord"] = f"{coord_s:.4f}"
            if rep_s   is not None: postfix["rep"]   = f"{rep_s:.4f}"
            if cov_s   is not None: postfix["cov"]   = f"{cov_s:.4f}"
            if vel_s   is not None: postfix["vel"]   = f"{vel_s:.4f}"
            if skipped_steps > 0:   postfix["skip"]  = skipped_steps
            pbar.set_postfix(**postfix)

        # ── Epoch-end metrics ─────────────────────────────────────────────────
        if skipped_steps > 0:
            print(f"  ⚠ {skipped_steps} optimizer steps skipped due to NaN/Inf gradients.")
        train_ce    = ce_total    / max(n_steps, 1)
        train_coord = coord_total / n_coord if n_coord > 0 else None
        train_rep   = rep_total   / n_rep   if n_rep   > 0 else None
        train_cov   = cov_total   / n_cov   if n_cov   > 0 else None
        train_vel   = vel_total   / n_vel   if n_vel   > 0 else None

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
            max_seq_len=args.max_seq_len,
            coord_stats=coord_stats,
            step_weight_decay=args.step_weight_decay,
            rep_min_dist=args.rep_min_dist,
            rep_lambda=args.rep_lambda,
            coverage_lambda=args.coverage_lambda,
            velocity_lambda=args.velocity_lambda,
        )
        val_ce    = val_metrics["ce_loss"]
        val_coord = val_metrics["coord_loss"]
        val_rep   = val_metrics["rep_loss"]
        val_cov   = val_metrics["cov_loss"]
        val_vel   = val_metrics["vel_loss"]

        # Combined metric for checkpoint selection
        val_combined = val_ce
        if val_combined is not None:
            if val_coord is not None:
                val_combined += args.coord_lambda * val_coord
            if val_rep is not None:
                val_combined += args.rep_lambda * val_rep
            if val_cov is not None:
                val_combined += args.coverage_lambda * val_cov
            if val_vel is not None:
                val_combined += args.velocity_lambda * val_vel

        # ── Generation eval ───────────────────────────────────────────────
        gen_metrics: dict = {}
        if args.gen_eval_epochs > 0 and (epoch + 1) % args.gen_eval_epochs == 0:
            if val_rows:
                vstart = 0 if args.gen_eval_fixed_slice else (
                    ((args.first_epoch - 1 + epoch) * args.gen_eval_samples) % len(val_rows)
                )
                gen_eval_rows = _gen_eval_slice(val_rows, args.gen_eval_samples, vstart)
                print(f"  gen_eval val slice start={vstart} (epoch {epoch_display})")
                gen_samples = [BubbleViewDataset(gen_eval_rows, clicks_only=args.clicks_only)[i]
                               for i in range(len(gen_eval_rows))]
                gen_metrics = generation_eval(
                    model, gen_samples, processor, device, args.gen_max_new_tokens,
                    system_prompt=args.system_prompt,
                    temperature=args.gen_temperature,
                    repetition_penalty=args.gen_repetition_penalty,
                    block_repeats=args.gen_block_repeats,
                    block_repeat_penalty=args.gen_block_repeat_penalty,
                )

        # ── Logging ───────────────────────────────────────────────────────
        parts = [
            f"epoch {epoch_display}",
            f"train_ce={train_ce:.4f}",
            f"val_ce={val_ce:.4f}" if val_ce is not None else "val_ce=—",
        ]
        if train_coord is not None: parts.append(f"train_coord={train_coord:.4f}")
        if val_coord   is not None: parts.append(f"val_coord={val_coord:.4f}")
        if train_rep   is not None: parts.append(f"train_rep={train_rep:.4f}")
        if val_rep     is not None: parts.append(f"val_rep={val_rep:.4f}")
        if train_cov   is not None: parts.append(f"train_cov={train_cov:.4f}")
        if val_cov     is not None: parts.append(f"val_cov={val_cov:.4f}")
        if train_vel   is not None: parts.append(f"train_vel={train_vel:.4f}")
        if val_vel     is not None: parts.append(f"val_vel={val_vel:.4f}")
        if gen_metrics:
            parts.append(f"gen_dtw={gen_metrics['gen_dtw']:.2f}px")
            parts.append(f"gen_len_err={gen_metrics['gen_len_err']:.3f}")
        print("  ".join(parts))

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_combined is not None and not math.isnan(val_combined):
            if best_val is None or val_combined < best_val:
                best_val = val_combined
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                with open(best_dir / "best_metrics.json", "w") as f:
                    json.dump({
                        "epoch": epoch_display,
                        "val_ce": val_ce, "val_coord": val_coord,
                        "val_rep": val_rep, "val_cov": val_cov,
                        "val_vel": val_vel,
                        "val_combined": val_combined,
                        "train_ce": train_ce, "train_coord": train_coord,
                        "train_rep": train_rep, "train_cov": train_cov,
                        "train_vel": train_vel,
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
