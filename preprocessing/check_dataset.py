import json
import random
from PIL import Image

with open("data/vcot_dataset.json") as f:
    data = json.load(f)

print("Total samples:", len(data))

# Check random samples
for i in range(5):
    sample = random.choice(data)

    print("\n--- SAMPLE ---")
    print("Prompt:", sample["prompt"])
    print("Target:", sample["target"])

    # Check image loads
    try:
        img = Image.open(sample["image"])
        print("Image size:", img.size)
    except:
        print("❌ Image failed to load:", sample["image"])