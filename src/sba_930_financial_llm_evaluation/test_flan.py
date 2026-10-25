import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

print("Loading FLAN-T5 Small...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Use Apple Silicon acceleration if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded successfully on {device}!")

# Financial analysis test prompt
prompt = (
    "Explain in one sentence how rising inflation "
    "can affect the stock market."
)

# Convert text into model inputs
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
).to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

# Decode model output into text
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print("\nFinancial Analysis Prompt:")
print(prompt)

print("\nFLAN-T5 Response:")
print(response)
