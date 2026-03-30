import json
import random
import sys
from pathlib import Path

from PIL import Image

_PRE = Path(__file__).resolve().parent
if str(_PRE) not in sys.path:
    sys.path.insert(0, str(_PRE))
from vcot_target import parse_vcot_target

with open("data/vcot_dataset.json") as f:
    data = json.load(f)

print("Total samples:", len(data))

# Check random samples
for i in range(5):
    sample = random.choice(data)

    print("\n--- SAMPLE ---")
    print("Prompt:", sample["prompt"])
    clicks, answer = parse_vcot_target(sample["target"])
    print("Target (full):", sample["target"])
    print("Target (clicks part):", clicks[:120] + ("..." if len(clicks) > 120 else ""))
    print("Target (answer after <sep>):", repr(answer))

    # Check image loads
    try:
        img = Image.open(sample["image"])
        print("Image size:", img.size)
    except:
        print("❌ Image failed to load:", sample["image"])