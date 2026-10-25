
from pathlib import Path
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("artifacts/forecasts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load historical and evaluation datasets
spy_history = pd.read_csv(DATA_DIR / "spy_history.csv")
spy_test = pd.read_csv(DATA_DIR / "spy_test.csv")

cpi_history = pd.read_csv(DATA_DIR / "cpi_history.csv")
cpi_test = pd.read_csv(DATA_DIR / "cpi_test.csv")

# Sort observations
spy_history = spy_history.sort_values("Date")
spy_test = spy_test.sort_values("Date")
cpi_history = cpi_history.sort_values("observation_date")
cpi_test = cpi_test.sort_values("observation_date")

# Historical inputs
last_spy = float(spy_history["Close"].iloc[-1])
last_inflation = float(cpi_history["inflation_rate"].iloc[-1])

# Actual outcomes (used ONLY for scoring)
january_spy = spy_test[
    spy_test["Date"].str.startswith("2024-01")
]
actual_spy = float(january_spy["Close"].iloc[-1])

january_cpi = cpi_test[
    cpi_test["observation_date"].str.startswith("2024-01")
]
actual_inflation = float(january_cpi["inflation_rate"].iloc[0])

# Forecast prompts
prompts = {
    "spy_price": (
        f"SPY closed at ${last_spy:.2f} at the end of "
        "December 2023. Predict its closing price "
        "at the end of January 2024. "
        "Return only one number."
    ),
    "inflation_rate": (
        f"US annual CPI inflation was {last_inflation:.2f}% "
        "in December 2023. Predict annual CPI inflation "
        "for January 2024. Return only one number."
    ),
}

actual_values = {
    "spy_price": actual_spy,
    "inflation_rate": actual_inflation,
}

models = [
    "google/flan-t5-small",
    "google/flan-t5-base",
]

device = "mps" if torch.backends.mps.is_available() else "cpu"
results = []

def extract_number(response):
    """Extract a number only if the response is numeric."""
    match = re.fullmatch(
        r"\s*[$]?\s*(-?\d+(?:\.\d+)?)\s*%?\s*",
        response,
    )
    return float(match.group(1)) if match else None


for model_name in models:
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    for task, prompt in prompts.items():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
            )

        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        prediction = extract_number(response)
        actual = actual_values[task]

        error = (
            abs(prediction - actual)
            if prediction is not None
            else None
        )

        results.append({
            "model": model_name,
            "task": task,
            "prompt": prompt,
            "raw_response": response,
            "prediction": prediction,
            "actual": actual,
            "absolute_error": error,
        })

        print(f"\nTask: {task}")
        print(f"Response: {response}")
        print(f"Parsed prediction: {prediction}")
        print(f"Actual value: {actual:.4f}")
        print(f"Absolute error: {error}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "numerical_forecasts.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print(results_df[
    ["model", "task", "prediction", "actual", "absolute_error"]
].to_string(index=False))

