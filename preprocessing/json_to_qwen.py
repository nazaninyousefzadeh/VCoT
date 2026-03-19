import json

INPUT = "data/vcot_dataset.json"
OUTPUT = "data/qwen_dataset.json"

with open(INPUT) as f:
    data = json.load(f)

converted = []

for sample in data:

    converted.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["prompt"]}
                ]
            },
            {
                "role": "assistant",
                "content": sample["target"]
            }
        ]
    })

with open(OUTPUT, "w") as f:
    json.dump(converted, f)

print("Saved to", OUTPUT)