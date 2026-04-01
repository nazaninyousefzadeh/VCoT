"""Streamlit dashboard — Baseline vs Saliency Overlay comparison."""
import csv
import json
import re
from pathlib import Path

import streamlit as st
from PIL import Image

BASELINE_FILE  = "data/qwen_responses_unique.json"
SALIENCY_FILE  = "data/qwen_responses_saliency.json"
METADATA_CSV   = "data/SalChartQA/unified_approved.csv"
ASSISTANT_MARKER = "\nassistant\n"


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_answer(response: str | None) -> str:
    if not response:
        return "(no response)"
    if ASSISTANT_MARKER in response:
        return response.rsplit(ASSISTANT_MARKER, 1)[-1].strip()
    return response.strip()


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
def load_all():
    # metadata lookup
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


# ── load ─────────────────────────────────────────────────────────────────────
st.title("VCoT — Qwen Response Explorer")
baseline, sal_idx = load_all()

# ── sidebar filters (shared) ─────────────────────────────────────────────────
st.sidebar.header("Filters")
all_img_types = sorted({r["image_type"] for r in baseline})
all_q_types   = sorted({r["question_type"] for r in baseline})

img_filter    = st.sidebar.multiselect("Chart type", all_img_types, default=all_img_types)
q_filter      = st.sidebar.multiselect("Question type", all_q_types, default=all_q_types)
simple_filter = st.sidebar.radio("Chart complexity", ["All", "Simple", "Non-simple"], horizontal=True)

def pass_simple(r):
    if simple_filter == "All":      return True
    if simple_filter == "Simple":   return r["is_simple"] is True
    return r["is_simple"] is False

filtered = [r for r in baseline
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
q_options = {f"[{r['index']}] {r['prompt'][:70]}": r for r in questions_for_img}
sel_q_label = st.sidebar.selectbox("Question", list(q_options.keys()))
base_row = q_options[sel_q_label]
sal_row  = sal_idx.get(base_row["index"])


# ── tab renderer ─────────────────────────────────────────────────────────────
def render_tab(row: dict, label: str):
    if row is None:
        st.warning("No data available for this condition.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Chart")
        img_path = Path(row["image"])
        if img_path.exists():
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.warning(f"Image not found: {img_path}")
        simple_tag = {True: "simple", False: "non-simple", None: "unknown"}[row["is_simple"]]
        st.caption(f"Type: `{row['image_type']}`  ·  `{simple_tag}`")

    with col2:
        st.subheader("Question")
        st.write(row["prompt"])
        st.caption(f"Question type: `{row['question_type']}`")
        ok_btn = st.button("Submit to Qwen", key=f"btn_{label}")

    if ok_btn:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Qwen Response")
            st.write(extract_answer(row.get("response")))
        with c2:
            st.subheader("Ground Truth")
            st.write(row.get("ground_truth_answer", "—"))
        correct = is_correct(row.get("response"), row.get("ground_truth_answer"))
        if correct:
            st.success("Correct ✓")
        else:
            st.error("Incorrect ✗")

    st.divider()
    st.subheader("Examples from current filter")
    data_pool = filtered if label == "baseline" else [sal_idx[r["index"]] for r in filtered if r["index"] in sal_idx]
    correct_ex = next((r for r in data_pool if is_correct(r.get("response"), r.get("ground_truth_answer"))), None)
    wrong_ex   = next((r for r in data_pool if not is_correct(r.get("response"), r.get("ground_truth_answer")) and r.get("response")), None)

    ex_col1, ex_col2 = st.columns(2)
    for col, ex, ex_label, color in [
        (ex_col1, correct_ex, "Correct example", "green"),
        (ex_col2, wrong_ex,   "Incorrect example", "red"),
    ]:
        with col:
            st.markdown(f"**:{color}[{ex_label}]**")
            if ex:
                p = Path(ex["image"])
                if p.exists():
                    st.image(Image.open(p), use_container_width=True)
                simple_tag = {True: "simple", False: "non-simple", None: "?"}[ex["is_simple"]]
                st.caption(f"{ex['prompt']}  |  `{ex['image_type']}` / `{ex['question_type']}` / `{simple_tag}`")
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
        sal_correct  = is_correct(sal_row.get("response"),  sal_row.get("ground_truth_answer"))
        same_response = base_row.get("response") == sal_row.get("response")

        if same_response:
            st.info("Same response as baseline for this sample.")
        elif base_correct and not sal_correct:
            st.warning("Saliency hurt this sample — baseline was correct, saliency is wrong.")
        elif not base_correct and sal_correct:
            st.success("Saliency helped this sample — baseline was wrong, saliency is correct.")
        else:
            st.info("Both conditions wrong, but with different responses.")

    render_tab(sal_row, "saliency")
