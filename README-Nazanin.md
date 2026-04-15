# VCoT

This project explores fine-tuning an open-source Large Vision-Language Model (LVLMs) to predict sequential human click patterns. We use the SalChartQA dataset for fine-tuning, aiming to model how humans visually navigate and interact with chart-based visual content in a step-by-step manner. The project is built with Python 3.10.

## Training scripts (quick trace)

Run all commands from the repository root. Each script accepts `--help` for full arguments (data paths, epochs, loss weights, etc.).

| What you are training | Script to run |
|----------------------|----------------|
| **CE + soft-DTW** (cross-entropy plus spatial soft-DTW on click coordinates) | `python scripts/finetune_qwen.py` |
| **Auxiliary losses** (extended setup: soft-DTW on positions, velocity / delta matching, logits processors, etc.) | `python scripts/training_modules/train_auxilary_updated.py` |
| **Point head** (LoRA + lightweight coordinate head on hidden states) | `python scripts/training_modules/train_point_head.py` |

**Where outputs go**

- **Saved models and run artifacts** (LoRA adapters, processors, `train_config.json`, epoch folders, `point_head.pt` when applicable) default under **`runs/`**:
  - CE+DTW and auxiliary training default to `--output_dir runs/qwen_lora` (use a different `--output_dir` per experiment so runs do not overwrite each other).
  - Point-head training defaults to `--output_dir runs/point_head_models`.
- **Data** live under **`data/`** (processed JSON such as `vcot_dataset_unique.json`). **Preprocessing outputs, checkpoints, eval JSON, and other experiment outputs** also accumulate under **`runs/`** (e.g. per-experiment subfolders like `runs/qwen_lora_v3/`, `runs/point_head_2ep/`).

Together, **`data/`** holds inputs; **`runs/`** holds anything produced by training or evaluation for easy tracing.

## Setup

This project uses a virtual environment:

- **`venv/`** — Qwen2-VL (`requirements.txt`): `pip install -r requirements.txt`


## Data

`data/vcot_dataset.json` rows use `target` as `"{click_tokens} | <sep> | {answer}"`: the **answer** is the substring **after** ` | <sep> | `; before it is the `<click>x,y</click>` sequence. Helpers: `preprocessing/vcot_target.py`.

**Preprocessing pipeline**

1. `python preprocessing/build_dataset.py` → full dataset (multiple trajectories per chart–question).
2. `python preprocessing/dedupe_by_image_prompt.py` → `data/vcot_dataset_unique.json`: **one row per unique `(image, prompt)`** (first trajectory kept). Same image with **different** questions stays multiple rows.

SalChartQA:
- `unified_approved.csv` — aggregate counts for each participant by question
- `fixationByVis/` — clicks and timestamps per question (columns: x, y, time)