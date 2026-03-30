# VCoT

This project explores fine-tuning two open-source Large Vision-Language Models (LVLMs) to predict sequential human click patterns. We use the SalChartQA dataset for fine-tuning, aiming to model how humans visually navigate and interact with chart-based visual content in a step-by-step manner. The project is built with Python 3.10. This README will be updated as the project evolves.


## Setup

This project uses two virtual environments due to dependency conflicts between models:

- **`venv/`** — Qwen2-VL (`requirements.txt`): `pip install -r requirements.txt`
- **`venv_tinychart/`** — TinyChart (`requirements_tinychart.txt`): `pip install -r requirements_tinychart.txt`

## Data

`data/vcot_dataset.json` rows use `target` as `"{click_tokens} | <sep> | {answer}"`: the **answer** is the substring **after** ` | <sep> | `; before it is the `<click>x,y</click>` sequence. Helpers: `preprocessing/vcot_target.py`.

**Preprocessing pipeline**

1. `python preprocessing/build_dataset.py` → full dataset (multiple trajectories per chart–question).
2. `python preprocessing/dedupe_by_image_prompt.py` → `data/vcot_dataset_unique.json`: **one row per unique `(image, prompt)`** (first trajectory kept). Same image with **different** questions stays multiple rows.

SalChartQA:
- `unified_approved.csv` — aggregate counts for each participant by question
- `fixationByVis/` — clicks and timestamps per question (columns: x, y, time)