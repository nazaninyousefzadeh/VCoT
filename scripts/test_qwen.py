import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

IMAGE_PATH = "data/SalChartQA/raw_img/00006834003065.png"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Loading model: {MODEL_NAME}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")
print(f"Model device: {model.device}")

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {image.size}")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What do you see in this chart?"}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

print("Generating response...")
output = model.generate(**inputs, max_new_tokens=128)
response = processor.batch_decode(output, skip_special_tokens=True)[0]

print(f"\nResponse:\n{response}")
print("\nQwen2-VL test passed!")
