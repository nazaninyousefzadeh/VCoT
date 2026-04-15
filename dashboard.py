"""Streamlit dashboard — Baseline vs Saliency Overlay comparison."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.plot_click_arrows import (  # noqa: E402
    parse_clicks_from_target,
    plot_click_arrows,
    plot_click_arrows_compare,
)

BASELINE_FILE  = "data/qwen_responses_unique.json"
SALIENCY_FILE  = "data/qwen_responses_saliency.json"
METADATA_CSV   = "data/SalChartQA/unified_approved.csv"
# Point head vs Auxiliary: separate qa_answers.json from infer_qa_with_pred_clicks
DEFAULT_POINT_HEAD_QA = "runs/test_preds_two_models/point_head_2ep/qa_answers.json"
DEFAULT_AUXILIARY_QA = "runs/test_preds_two_models/qwen_lora_v3/qa_answers.json"
# Merged text answers (fallback + Qwen+GT-clicks column), keyed by qa_index
DEFAULT_MERGED_QA = "runs/test_preds_two_models/all_three_qa_merged_unique.json"

ASSISTANT_MARKER = "\nassistant\n"
_RE_CLICK = re.compile(r"<click>(\d+),(\d+)</click>")


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_answer(response: str | None) -> str:
    if not response:
        return "(no response)"
    if ASSISTANT_MARKER in response:
        return response.rsplit(ASSISTANT_MARKER, 1)[-1].strip()
    return response.strip()


def parse_clicks_from_response(response: str | None) -> list[tuple[float, float]]:
    """`<click>x,y</click>` tags in the full message or assistant tail (0–1000 normalized)."""
    if not response:
        return []
    text = extract_answer(response) if ASSISTANT_MARKER in (response or "") else (response or "")
    return [(float(x), float(y)) for x, y in _RE_CLICK.findall(text)]


def format_clicks_norm(clicks: list[tuple[float, float]]) -> str:
    if not clicks:
        return "(none)"
    return " → ".join(f"({int(round(x))}, {int(round(y))})" for x, y in clicks)


def clicks_pixels_to_norm1000(
    pairs: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if not pairs or width <= 0 or height <= 0:
        return []
    out: list[tuple[float, float]] = []
    for pair in pairs:
        x, y = float(pair[0]), float(pair[1])
        out.append((x / width * 1000.0, y / height * 1000.0))
    return out


def pred_clicks_norm_from_qa(extra: dict | None, width: int, height: int) -> list[tuple[float, float]]:
    """
    Build 0–1000 coordinates for plotting from ``qa_answers.json`` ``pred_clicks_pixel`` / ``clicks_pixel``.

    Some eval exports store **0–1000 normalized** pairs in that field (same convention as the dataset).
    If any value is larger than the raw image width/height, we treat the sequence as normalized;
    otherwise we treat it as **pixel** coordinates and scale by ``width``/``height``.
    """
    if not extra:
        return []
    px = extra.get("clicks_pixel")
    if not isinstance(px, list) or not px:
        return []
    if width <= 0 or height <= 0:
        return []
    pairs: list[tuple[float, float]] = [(float(p[0]), float(p[1])) for p in px]
    max_x = max(x for x, _ in pairs)
    max_y = max(y for _, y in pairs)
    if max_x > width + 0.5 or max_y > height + 0.5:
        return pairs
    return clicks_pixels_to_norm1000(px, width, height)


_VCOT_SEP = " | <sep> | "


def split_vcot_target(target: str) -> tuple[str, str]:
    """Split ``target`` into click prefix and answer suffix (dataset / merged JSON)."""
    t = (target or "").strip()
    if _VCOT_SEP in t:
        left, right = t.split(_VCOT_SEP, 1)
        return left.strip(), right.strip()
    return t, ""


def ground_truth_click_string(row: dict) -> str:
    s = (row.get("ground_truth_clicks") or "").strip()
    if s:
        return s
    gt = (row.get("ground_truth") or "").strip()
    if _VCOT_SEP in gt:
        return split_vcot_target(gt)[0]
    if "|" in gt:
        return gt.split("|", 1)[0].strip()
    return gt


_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _word_numbers_in_text(text: str) -> list[float]:
    return [float(_WORD_TO_NUM[w]) for w in re.findall(r"\b[a-z]+\b", text.lower()) if w in _WORD_TO_NUM]


def _token_set(s: str) -> set[str]:
    return {
        t.strip(".,;:\"'()!?")
        for t in re.split(r"[\s,&]+", s.lower())
        if t.strip(".,;:\"'()!?") and t.strip(".,;:\"'()!?") not in ("and", "or", "the")
    }


_NO_PHRASES = [
    "does not", "do not", "did not", "is not", "are not", "was not", "were not",
    "cannot", "can not", "could not", "would not", "will not", "should not",
    "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "couldn't", "wouldn't", "won't", "shouldn't",
    "no", "incorrect", "false", "never", "neither", "none",
]
_YES_PHRASES = ["yes", "correct", "true", "affirmative", "indeed"]


def _canonical_yes_no(text: str) -> str | None:
    t = text.lower().strip()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", t) if s.strip()]
    conclusion = sentences[-1] if sentences else t
    for scope in (conclusion, t):
        if any(p in scope for p in _NO_PHRASES):
            return "no"
        if any(p in scope for p in _YES_PHRASES):
            return "yes"
    hits = list(re.finditer(r"\b(yes|no)\b", t))
    return hits[-1].group(1) if hits else None


def _ngrams_from_text(text: str, n: int) -> list[str]:
    words = re.split(r"\s+", text.strip())
    if n == 1:
        return [w.strip(".,;:\"'()").lower() for w in words]
    return [
        " ".join(words[i : i + n]).strip(".,;:\"'()").lower()
        for i in range(len(words) - n + 1)
    ]


def _fuzzy_match(gt: str, pred: str, threshold: float = 0.85) -> bool:
    from difflib import SequenceMatcher
    if len(gt) < 4:
        return False
    gt_words = gt.split()
    n = len(gt_words)
    for candidate in _ngrams_from_text(pred, n):
        if abs(len(candidate) - len(gt)) <= n:
            if SequenceMatcher(None, gt, candidate).ratio() >= threshold:
                return True
    return False


def is_correct(response: str | None, ground_truth: str | None) -> bool:
    if not response or not ground_truth:
        return False
    pred = extract_answer(response).lower().strip()
    gt = ground_truth.lower().strip()
    try:
        gt_num = float(gt.replace(",", "").rstrip("%"))
        for n in re.findall(r"-?[\d,]+\.?\d*%?", pred):
            try:
                if abs(float(n.replace(",", "").rstrip("%")) - gt_num) / max(abs(gt_num), 1e-9) <= 0.05:
                    return True
            except ValueError:
                pass
        # word-number fallback: GT is "1" but model says "one"
        for pv in _word_numbers_in_text(pred):
            if abs(pv - gt_num) / max(abs(gt_num), 1e-9) <= 0.05:
                return True
        return False
    except ValueError:
        pass
    if gt in ("yes", "no"):
        return _canonical_yes_no(pred) == gt
    if pred == gt:
        return True
    gt_tokens = _token_set(gt)
    pred_tokens = _token_set(pred)
    if len(gt_tokens) > 1 and (gt_tokens == pred_tokens or gt_tokens.issubset(pred_tokens)):
        return True
    if re.search(r"\b" + re.escape(gt) + r"\b", pred):
        return True

    # fuzzy match: 1-char typos and multi-word GTs (e.g. "South Korea")
    if _fuzzy_match(gt, pred):
        return True

    return False


def _normalize_img_name(name: str) -> str:
    """Strip saliency overlay suffix: '797_Q1_overlay.png' -> '797.png'"""
    return re.sub(r"_Q\d+_overlay", "", name)


@st.cache_data
def load_metadata_lookup() -> dict[tuple[str, str], tuple[str, str, bool]]:
    lookup: dict[tuple[str, str], tuple[str, str, bool]] = {}
    with open(METADATA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            k = (row["image_name"].strip(), row["question"].strip())
            if k not in lookup:
                lookup[k] = (
                    row["image_type"].strip(),
                    row["question_type"].strip(),
                    row["is_chart_simple"].strip() == "True",
                )
    return lookup


@st.cache_data
def load_all():
    lookup = load_metadata_lookup()

    def attach_meta(data):
        for r in data:
            img_name = _normalize_img_name(Path(r["image"]).name)
            it, qt, simple = lookup.get((img_name, r["prompt"].strip()), ("unknown", "unknown", None))
            r["image_type"] = it
            r["question_type"] = qt
            r["is_simple"] = simple
        return data

    baseline = attach_meta(json.load(open(BASELINE_FILE)))
    saliency_raw = json.load(open(SALIENCY_FILE)) if Path(SALIENCY_FILE).exists() else []
    saliency = attach_meta(saliency_raw)

    # index saliency by (original image stem + prompt) for cross-tab lookup
    sal_idx: dict[int, dict] = {r["index"]: r for r in saliency}

    return baseline, sal_idx


@st.cache_data
def load_qa_answers_json(path: str) -> dict[int, dict]:
    """``qa_answers.json`` from infer_qa_with_pred_clicks: pred clicks + answer per index."""
    p = Path(path.strip()) if path else Path()
    if not path.strip() or not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for s in raw.get("per_sample", []):
        idx = int(s["index"])
        px = s.get("pred_clicks_pixel") or s.get("clicks_pixel") or []
        out[idx] = {
            "image": (s.get("image") or "").strip(),
            "prompt": (s.get("prompt") or "").strip(),
            "clicks_pixel": px,
            "answer": s.get("answer"),
            "click_source": s.get("click_source", ""),
        }
    return out


@st.cache_data
def load_optional_merged_qa(path: str) -> dict[int, dict]:
    """Merged list JSON: ``qa_index`` → extra text answers (e.g. pointhead vs Qwen vs GT-clicks QA)."""
    p = Path(path.strip()) if path else Path()
    if not path.strip() or not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    return {int(r["qa_index"]): r for r in data if "qa_index" in r}


# ── load ─────────────────────────────────────────────────────────────────────
st.title("VCoT — Qwen Response Explorer")
baseline, sal_idx = load_all()

_default_ph = DEFAULT_POINT_HEAD_QA if (_REPO / DEFAULT_POINT_HEAD_QA).is_file() else ""
_default_aux = DEFAULT_AUXILIARY_QA if (_REPO / DEFAULT_AUXILIARY_QA).is_file() else ""
_default_merged = DEFAULT_MERGED_QA if (_REPO / DEFAULT_MERGED_QA).is_file() else ""

# ── sidebar: QA sources (drives which ~597 rows appear in Plot / Question) ───
st.sidebar.header("Click + QA exports")
ph_path = st.sidebar.text_input(
    "Point head — qa_answers.json",
    value=_default_ph,
    help="Pred clicks from Point_head eval + QA answer. Intersected with Auxiliary below.",
)
aux_path = st.sidebar.text_input(
    "Auxiliary (LoRA v3) — qa_answers.json",
    value=_default_aux,
    help="Pred clicks from Auxiliary-loss run. Only indices present in **both** files are selectable.",
)
merged_path = st.sidebar.text_input(
    "Merged answers JSON (fallback)",
    value=_default_merged,
    help="all_three_qa_merged_unique.json — fills answers if QA JSONs missing; adds Qwen+GT-clicks column.",
)
ph_by_idx = load_qa_answers_json(ph_path)
aux_by_idx = load_qa_answers_json(aux_path)
merged_by_idx = load_optional_merged_qa(merged_path)

both_idx = set(ph_by_idx.keys()) & set(aux_by_idx.keys())
if not both_idx:
    st.error(
        "Set valid paths to **both** Point head and Auxiliary `qa_answers.json` files. "
        "They must share at least one sample index."
    )
    st.stop()

# Eval JSON indices (0–596) are NOT the same as rows in ``qwen_responses_unique.json`` (0–2799).
# Join baseline/saliency by (image, prompt); use ``qa_eval_index`` for QA JSON + merged lookups.
meta_lookup = load_metadata_lookup()
key_to_baseline = {(r["image"].strip(), r["prompt"].strip()): r for r in baseline}
sal_by_key: dict[tuple[str, str], dict] = {}
for _sr in sal_idx.values():
    _k = (_sr["prompt"].strip(), _normalize_img_name(Path(_sr["image"]).name))
    sal_by_key[_k] = _sr

ph_aux_mismatch: list[int] = []
skipped_no_merged: list[int] = []
baseline_scoped: list[dict] = []

for eval_idx in sorted(both_idx):
    ph = ph_by_idx[eval_idx]
    au = aux_by_idx[eval_idx]
    if ph.get("image") != au.get("image") or ph.get("prompt") != au.get("prompt"):
        ph_aux_mismatch.append(eval_idx)
    img, pr = ph["image"].strip(), ph["prompt"].strip()
    k = (img, pr)
    br = key_to_baseline.get(k)
    mr = merged_by_idx.get(eval_idx)
    sk = (pr, _normalize_img_name(Path(img).name))
    sal_r = sal_by_key.get(sk)

    if br:
        row = {**br, "qa_eval_index": eval_idx, "_sal_row": sal_r}
    else:
        if not mr:
            skipped_no_merged.append(eval_idx)
            continue
        gt_target = mr.get("target") or ""
        clicks_part, ans_part = split_vcot_target(gt_target)
        img_name = _normalize_img_name(Path(img).name)
        it, qt, simple = meta_lookup.get((img_name, pr), ("unknown", "unknown", None))
        row = {
            "qa_eval_index": eval_idx,
            "index": None,
            "image": img,
            "prompt": pr,
            "response": None,
            "ground_truth": gt_target,
            "ground_truth_clicks": clicks_part if "<click>" in clicks_part else "",
            "ground_truth_answer": ans_part,
            "image_type": it,
            "question_type": qt,
            "is_simple": simple,
            "_sal_row": sal_r,
        }
    baseline_scoped.append(row)

if not baseline_scoped:
    st.error(
        "No samples could be built. Load **merged answers JSON** (default path) so rows not in "
        "`qwen_responses_unique.json` still get ground-truth clicks/answers from `target`."
    )
    st.stop()

if ph_aux_mismatch:
    st.warning(f"Point head vs Auxiliary image/prompt differ for **{len(ph_aux_mismatch)}** eval indices (showing first 5): {ph_aux_mismatch[:5]}")

if skipped_no_merged:
    st.warning(
        f"**{len(skipped_no_merged)}** eval rows have no matching baseline row and no merged JSON row — skipped. "
        "Set the merged JSON path to include all 597 test samples."
    )

n_ds = sum(1 for r in baseline_scoped if r.get("index") is not None)
st.caption(
    f"**{len(baseline_scoped)}** test samples (eval 0–596) with Point head + Auxiliary QA · "
    f"**{n_ds}** with Qwen responses in `qwen_responses_unique.json` · "
    f"**{len(baseline_scoped) - n_ds}** from merged `target` only."
)

# ── sidebar filters (shared) ─────────────────────────────────────────────────
st.sidebar.header("Filters")
all_img_types = sorted({r["image_type"] for r in baseline_scoped})
all_q_types   = sorted({r["question_type"] for r in baseline_scoped})

img_filter    = st.sidebar.multiselect("Chart type", all_img_types, default=all_img_types)
q_filter      = st.sidebar.multiselect("Question type", all_q_types, default=all_q_types)
simple_filter = st.sidebar.radio("Chart complexity", ["All", "Simple", "Non-simple"], horizontal=True)

def pass_simple(r):
    if simple_filter == "All":      return True
    if simple_filter == "Simple":   return r["is_simple"] is True
    return r["is_simple"] is False

filtered = [r for r in baseline_scoped
            if r["image_type"] in img_filter
            and r["question_type"] in q_filter
            and pass_simple(r)]

if not filtered:
    st.warning("No samples match the selected filters.")
    st.stop()

# ── sidebar sample selector (shared) ─────────────────────────────────────────
st.sidebar.header("Select a sample")
all_images   = sorted({r["image"] for r in filtered})
image_labels = {Path(p).name: p for p in all_images}
sel_img_name = st.sidebar.selectbox("Plot / Image", list(image_labels.keys()))
sel_img_path = image_labels[sel_img_name]

questions_for_img = [r for r in filtered if r["image"] == sel_img_path]


def _question_option_label(r: dict) -> str:
    qe = r["qa_eval_index"]
    ds = r.get("index")
    prefix = f"[QA {qe}]"
    if ds is not None:
        prefix += f" [ds {ds}]"
    return f"{prefix} {r['prompt'][:60]}"


q_options = {_question_option_label(r): r for r in questions_for_img}
sel_q_label = st.sidebar.selectbox("Question", list(q_options.keys()))
base_row = q_options[sel_q_label]
qa_eval_idx = int(base_row["qa_eval_index"])
sal_row = base_row.get("_sal_row") or (
    sal_idx.get(base_row["index"]) if base_row.get("index") is not None else None
)
merged_row = merged_by_idx.get(qa_eval_idx)
ph_extra = ph_by_idx.get(qa_eval_idx)
aux_extra = aux_by_idx.get(qa_eval_idx)


def render_answer_card(title: str, answer: str | None, response: str | None, gt_answer: str | None):
    st.markdown(f"**{title}**")
    st.write(answer if answer is not None else "—")
    if gt_answer is not None:
        st.caption(f"Ground-truth answer: `{gt_answer}`")
    clicks = parse_clicks_from_response(response)
    if clicks:
        st.caption("Clicks parsed from model output (0–1000 norm)")
        st.code(format_clicks_norm(clicks))
    else:
        st.caption("No `<click>` tags in model output")
    if gt_answer is not None:
        ok = is_correct(response, gt_answer)
        st.write("✓ correct" if ok else "✗ incorrect")


# ── shared: chart, question, clicks, arrows ────────────────────────────────
col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Chart")
    img_path = Path(base_row["image"])
    pil = None
    if img_path.exists():
        pil = Image.open(img_path).convert("RGB")
        st.image(pil, use_container_width=True)
    else:
        st.warning(f"Image not found: {img_path}")
    simple_tag = {True: "simple", False: "non-simple", None: "unknown"}[base_row["is_simple"]]
    _ds = base_row.get("index")
    _ds_s = f"`{_ds}`" if _ds is not None else "—"
    st.caption(
        f"Type: `{base_row['image_type']}`  ·  `{simple_tag}`  ·  QA eval `{qa_eval_idx}`  ·  dataset index {_ds_s}"
    )

with col_b:
    st.subheader("Question")
    st.write(base_row["prompt"])
    st.caption(f"Question type: `{base_row['question_type']}`")

gt_str = ground_truth_click_string(base_row)
gt_clicks = parse_clicks_from_target(gt_str) if gt_str else []
w_img = h_img = 0
if pil is not None:
    w_img, h_img = pil.size

if pil is not None:
    ph_norm = pred_clicks_norm_from_qa(ph_extra, w_img, h_img)
    aux_norm = pred_clicks_norm_from_qa(aux_extra, w_img, h_img)
else:
    ph_norm = []
    aux_norm = []


def _ans_ph() -> str:
    if ph_extra and ph_extra.get("answer") is not None:
        return str(ph_extra["answer"])
    if merged_row:
        return str(merged_row.get("pointhead_answer", "—"))
    return "—"


def _ans_aux() -> str:
    if aux_extra and aux_extra.get("answer") is not None:
        return str(aux_extra["answer"])
    if merged_row:
        return str(merged_row.get("qwen_lora_v3_answer", "—"))
    return "—"


def _ans_gt_clicks_qa() -> str | None:
    if merged_row and merged_row.get("gt_clicks_answer") is not None:
        return str(merged_row["gt_clicks_answer"])
    return None


st.subheader("Answers (this sample)")
a1, a2, a3, a4, a5, a6 = st.columns(6)
with a1:
    st.markdown("**Baseline**")
    st.write(extract_answer(base_row.get("response")))
    if base_row.get("ground_truth_answer") is not None:
        st.caption("✓" if is_correct(base_row.get("response"), base_row.get("ground_truth_answer")) else "✗")
with a2:
    st.markdown("**Saliency**")
    if sal_row:
        st.write(extract_answer(sal_row.get("response")))
        if base_row.get("ground_truth_answer") is not None:
            st.caption("✓" if is_correct(sal_row.get("response"), base_row.get("ground_truth_answer")) else "✗")
    else:
        st.write("—")
with a3:
    st.markdown("**Ground-truth**")
    st.write(str(base_row.get("ground_truth_answer", "—")))
with a4:
    st.markdown("**Point head → QA**")
    st.write(_ans_ph())
with a5:
    st.markdown("**Auxiliary → QA**")
    st.write(_ans_aux())
with a6:
    st.markdown("**Qwen + GT clicks**")
    gtc = _ans_gt_clicks_qa()
    st.write(gtc if gtc is not None else "—")
    st.caption("QA conditioned on human click path")

st.subheader("Click paths as arrows (raw chart)")
st.caption(
    "Ground truth = dataset labels (0–1000). Point head / Auxiliary: JSON may store **pixels** "
    "(scaled by image size) or **0–1000** values; if values exceed image width/height, they are treated as 0–1000."
)

if pil is None:
    st.warning("Load a valid chart image to draw arrows.")
else:
    ar1, ar2, ar3 = st.columns(3)
    with ar1:
        st.markdown("**Ground truth**")
        st.code(gt_str or "(empty)", language="text")
        st.caption(format_clicks_norm(gt_clicks))
        if gt_clicks:
            fig_gt, _ = plot_click_arrows(pil, gt_clicks)
            st.pyplot(fig_gt)
            plt.close(fig_gt)
        else:
            st.info("No GT clicks.")
    with ar2:
        st.markdown("**Point head (predicted)**")
        if ph_extra:
            st.caption(format_clicks_norm(ph_norm) if ph_norm else "(none)")
            if ph_norm:
                fig_ph, _ = plot_click_arrows(
                    pil,
                    ph_norm,
                    arrow_color="#ff7f0e",
                    point_color="#ffe0b0",
                    point_edge="#aa4400",
                )
                st.pyplot(fig_ph)
                plt.close(fig_ph)
            else:
                st.info("No click pixels in JSON.")
        else:
            st.info("Load Point head QA JSON in the sidebar.")
    with ar3:
        st.markdown("**Auxiliary (predicted)**")
        if aux_extra:
            st.caption(format_clicks_norm(aux_norm) if aux_norm else "(none)")
            if aux_norm:
                fig_ax, _ = plot_click_arrows(
                    pil,
                    aux_norm,
                    arrow_color="#1f77b4",
                    point_color="#c8e4f5",
                    point_edge="#0d4a7a",
                )
                st.pyplot(fig_ax)
                plt.close(fig_ax)
            else:
                st.info("No click pixels in JSON.")
        else:
            st.info("Load Auxiliary QA JSON in the sidebar.")

    if ph_norm and aux_norm and gt_clicks:
        st.subheader("Overlay: GT vs both models")
        o1, o2 = st.columns(2)
        with o1:
            fig_c, _ = plot_click_arrows_compare(pil, ph_norm, gt_clicks)
            st.pyplot(fig_c)
            plt.close(fig_c)
            st.caption("Orange · GT · Cyan · Point head")
        with o2:
            fig_c2, _ = plot_click_arrows_compare(pil, aux_norm, gt_clicks)
            st.pyplot(fig_c2)
            plt.close(fig_c2)
            st.caption("Orange · GT · Cyan · Auxiliary")


# ── tab renderer (answers + examples) ─────────────────────────────────────────
def render_tab(row: dict | None, label: str):
    if row is None:
        st.warning("No data available for this condition.")
        return

    st.subheader("Answer" if label == "baseline" else "Answer (saliency image)")
    gt_ans = row.get("ground_truth_answer")
    render_answer_card(
        "Extracted answer",
        extract_answer(row.get("response")),
        row.get("response"),
        gt_ans,
    )
    st.divider()
    st.subheader("Examples from current filter")
    if label == "baseline":
        data_pool = filtered
    else:
        data_pool = []
        for r in filtered:
            sr = r.get("_sal_row") or (
                sal_idx.get(r["index"]) if r.get("index") is not None else None
            )
            if sr is not None:
                data_pool.append(sr)
    correct_ex = next((r for r in data_pool if is_correct(r.get("response"), r.get("ground_truth_answer"))), None)
    wrong_ex = next(
        (r for r in data_pool if not is_correct(r.get("response"), r.get("ground_truth_answer")) and r.get("response")),
        None,
    )

    ex_col1, ex_col2 = st.columns(2)
    for col, ex, ex_label, color in [
        (ex_col1, correct_ex, "Correct example", "green"),
        (ex_col2, wrong_ex, "Incorrect example", "red"),
    ]:
        with col:
            st.markdown(f"**:{color}[{ex_label}]**")
            if ex:
                p = Path(ex["image"])
                if p.exists():
                    st.image(Image.open(p), use_container_width=True)
                simple_tag_ex = {True: "simple", False: "non-simple", None: "?"}[ex["is_simple"]]
                st.caption(f"{ex['prompt']}  |  `{ex['image_type']}` / `{ex['question_type']}` / `{simple_tag_ex}`")
                st.write(f"**Qwen:** {extract_answer(ex.get('response'))}")
                st.write(f"**GT:** {ex.get('ground_truth_answer', '—')}")
            else:
                st.write("—")


# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Baseline", "Saliency Overlay"])
with tab1:
    render_tab(base_row, "baseline")
with tab2:
    if sal_row:
        base_correct = is_correct(base_row.get("response"), base_row.get("ground_truth_answer"))
        sal_correct = is_correct(sal_row.get("response"), sal_row.get("ground_truth_answer"))
        same_response = base_row.get("response") == sal_row.get("response")

        if same_response:
            st.info("Same response as baseline for this sample.")
        elif base_correct and not sal_correct:
            st.warning("Saliency hurt this sample — baseline was correct, saliency is wrong.")
        elif not base_correct and sal_correct:
            st.success("Saliency helped this sample — baseline was wrong, saliency is correct.")
        else:
            st.info("Both conditions wrong, but with different responses.")

        sal_img = Path(sal_row["image"])
        if sal_img.exists():
            st.subheader("Saliency overlay (model input)")
            st.image(Image.open(sal_img), use_container_width=True)

    render_tab(sal_row, "saliency")
