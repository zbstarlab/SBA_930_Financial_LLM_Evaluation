
from pathlib import Path
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("artifacts/forecasts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load historical and evaluation data
history = pd.read_csv(
    DATA_DIR / "gold_history.csv",
    parse_dates=["Date"],
).sort_values("Date")

test = pd.read_csv(
    DATA_DIR / "gold_test.csv",
    parse_dates=["Date"],
).sort_values("Date")

# Use the final closing price of December 2023
last_price = float(history["Close"].iloc[-1])

# Actual closing price on the last trading day of January 2024
january = test[
    (test["Date"].dt.year == 2024)
    & (test["Date"].dt.month == 1)
]

if january.empty:
    raise ValueError("No January 2024 GLD data found")

actual_price = float(january["Close"].iloc[-1])

print(f"December 2023 GLD price: ${last_price:.2f}")
print(f"Actual January 2024 price: ${actual_price:.2f}")

prompt = (
    f"GLD, a gold ETF, closed at ${last_price:.2f} "
    "at the end of December 2023. "
    "Predict its closing price at the end of January 2024. "
    "Return only one positive number in dollars."
)

models = [
    "google/flan-t5-small",
    "google/flan-t5-base",
]

device = "mps" if torch.backends.mps.is_available() else "cpu"

results = []

def parse_number(response):
    text = response.strip()

    match = re.fullmatch(
        r"\$?\s*(?:\d+(?:\.\d*)?|\.\d+)\s*",
        text,
    )

    if not match:
        return None

    return float(text.replace("$", "").strip())


for model_name in models:
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    prediction = parse_number(response)

    # GLD prices must be positive
    valid = prediction is not None and prediction > 0

    error = (
        abs(prediction - actual_price)
        if valid
        else None
    )

    results.append({
        "model": model_name,
        "raw_response": response,
        "prediction": prediction,
        "actual": actual_price,
        "absolute_error": error,
        "valid_prediction": valid,
    })

    print(f"Response: {response}")
    print(f"Parsed prediction: {prediction}")
    print(f"Valid prediction: {valid}")
    print(f"Absolute error: {error}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

# Add naive persistence baseline
results.append({
    "model": "Naive baseline",
    "raw_response": str(last_price),
    "prediction": last_price,
    "actual": actual_price,
    "absolute_error": abs(last_price - actual_price),
    "valid_prediction": True,
})

df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "gold_forecast_comparison.csv"
df.to_csv(output_path, index=False)

print("\nGold Forecast Comparison:")
print(df.to_string(index=False))

print(f"\nResults saved to: {output_path}")
