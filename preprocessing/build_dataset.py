import os
import json
import pandas as pd
from PIL import Image

# =========================
# PATHS (EDIT IF NEEDED)
# =========================

BASE_FIXATION = "data/SalChartQA/fixationByVis"
IMAGE_FOLDER = "data/SalChartQA/raw_img"
CSV_PATH = "data/SalChartQA/unified_approved.csv"
OUTPUT_PATH = "data/vcot_dataset.json"

MAX_CLICKS = 10  # truncate long sequences


# =========================
# LOAD MASTER CSV
# =========================

print("Loading master CSV...")
df = pd.read_csv(CSV_PATH)

# Filter approved + correct responses
df = df[(df["is_correct"] == True)] #& (df["approved"] == True)]

# Convert types for matching
df["participant_id"] = df["participant_id"].astype(str)
df["is_approved"] = df["is_approved"].astype(str)

print(f"Filtered dataset size: {len(df)}")


# =========================
# HELPERS
# =========================

def parse_path(path):
    """
    Extract metadata from file path
    """
    parts = path.split(os.sep)

    viz_id = parts[-4]
    question_id = parts[-3]

    if parts[-2] in ["True", "False"]:
        approved = parts[-2]
        filename = parts[-1]
    else:
        approved = None
        filename = parts[-1]

    participant_id = filename.replace(".csv", "")

    return viz_id, question_id, approved, participant_id


def load_clicks(csv_path):
    try:
        df = pd.read_csv(csv_path)

        # Check if empty
        if df.empty:
            return None

        # Check if it has at least 2 columns
        if df.shape[1] < 2:
            return None

        clicks = df.iloc[:, :2].values.tolist()

        # Extra safety: remove NaN rows
        clicks = [c for c in clicks if not pd.isna(c[0]) and not pd.isna(c[1])]

        if len(clicks) == 0:
            return None

        return clicks

    except Exception as e:
        return None


def normalize_clicks(clicks, image_path):
    img = Image.open(image_path)
    w, h = img.size

    norm = []

    for x, y in clicks:
        nx = int((x / w) * 1000)
        ny = int((y / h) * 1000)
        norm.append((nx, ny))

    return norm


def clicks_to_tokens(clicks):
    return " ".join(
        [f"<click>{x},{y}</click>" for x, y in clicks]
    )


def map_viz_to_image(viz_id):
    """
    Adjust this if filenames differ
    """
    # Most likely case:
    return f"{viz_id}.png"


# =========================
# BUILD DATASET
# =========================

print("Building dataset...")

samples = []
skipped = 0

for root, dirs, files in os.walk(BASE_FIXATION):

    for file in files:

        if not file.endswith(".csv"):
            continue

        path = os.path.join(root, file)
        viz_id, qid, approved, pid = parse_path(path)
       # print(f"Metadata - Viz: {viz_id}, QID: {qid}, Approved: {approved}, PID: {pid}")
        image_name = map_viz_to_image(viz_id)
        image_path = os.path.join(IMAGE_FOLDER, image_name)

        if not os.path.exists(image_path):
            skipped += 1
            continue

        # Match row in master CSV
        matches = df[
            (df["image_name"] == image_name) &
            (df["participant_id"] == pid) &
            (df["is_approved"] == str(approved))
        ]

        if len(matches) == 0:
            skipped += 1
            continue

        row = matches.iloc[0]

        question = row["question"]
        answer = str(row["answer"])

        # Load clicks
        clicks = load_clicks(path)

        if clicks is None:
            skipped += 1
            continue

        # Truncate long sequences
        clicks = clicks[:MAX_CLICKS]

        # Normalize
        clicks = normalize_clicks(clicks, image_path)

        # Convert to tokens
        click_tokens = clicks_to_tokens(clicks)

        target = click_tokens + " | <sep> | " + answer

        sample = {
            "image": image_path,
            "prompt": question,
            "target": target
        }

        samples.append(sample)


# =========================
# SAVE OUTPUT
# =========================

print(f"Total samples: {len(samples)}")
print(f"Skipped: {skipped}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(samples, f)

print(f"Saved dataset to {OUTPUT_PATH}")


# =========================
# DEBUG: PRINT EXAMPLES
# =========================

print("\nSample outputs:\n")

for i in range(min(3, len(samples))):
    print(json.dumps(samples[i], indent=2))