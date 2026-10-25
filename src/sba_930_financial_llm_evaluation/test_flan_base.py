
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

print("Loading FLAN-T5 Base...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"Model loaded successfully on {device}!")

# Use the same prompt as FLAN-T5 Small
prompt = (
    "Explain in one sentence how rising inflation "
    "can affect the stock market."
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print("\nFinancial Analysis Prompt:")
print(prompt)

print("\nFLAN-T5 Base Response:")
print(response)
