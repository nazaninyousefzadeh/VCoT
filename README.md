# VCoT

This project explores fine-tuning two open-source Large Vision-Language Models (LVLMs) to predict sequential human click patterns. We use the SalChartQA dataset for fine-tuning, aiming to model how humans visually navigate and interact with chart-based visual content in a step-by-step manner. The project is built with Python 3.10. This README will be updated as the project evolves.


## Setup

This project uses two virtual environments due to dependency conflicts between models:

- **`venv/`** — Qwen2-VL (`requirements.txt`): `pip install -r requirements.txt`
- **`venv_tinychart/`** — TinyChart (`requirements_tinychart.txt`): `pip install -r requirements_tinychart.txt`

## Data

SalChartQA:
- `unified_approved.csv` — aggregate counts for each participant by question
- `fixationByVis/` — clicks and timestamps per question (columns: x, y, time)