"""
Test TinyChart-3B-768 loading and inference.

Run with: venv_tinychart/bin/python scripts/test_tinychart.py

Uses the official tinychart package from mPLUG-DocOwl.
"""

import torch
from PIL import Image

from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import process_images, tokenizer_image_token
from tinychart.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from tinychart.conversation import conv_templates

IMAGE_PATH = "data/SalChartQA/raw_img/00006834003065.png"
MODEL_PATH = "mPLUG/TinyChart-3B-768"
MODEL_NAME = "TinyChart-3B-768"

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Loading model: {MODEL_PATH}")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name=MODEL_NAME,
    device="cpu",
    torch_dtype=torch.float16,
)

device = "cpu"
model = model.to(device).float()

print("Model loaded successfully!")
print(f"Model device: {device}")

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {image.size} , {IMAGE_PATH}")

image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)

question = "What do you see in this chart?"

conv = conv_templates["phi"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)

print("Generating response...")
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=128,
        use_cache=True,
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nResponse:\n{response}")
print("\nTinyChart test passed!")
