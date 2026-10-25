
from pathlib import Path
import re

import pandas as pd

DATA_DIR = Path("data/processed")
FORECAST_DIR = Path("artifacts/forecasts")
OUTPUT_DIR = Path("artifacts/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load historical datasets
spy_history = pd.read_csv(DATA_DIR / "spy_history.csv")
cpi_history = pd.read_csv(DATA_DIR / "cpi_history.csv")

# Load LLM numerical forecasts
results = pd.read_csv(
    FORECAST_DIR / "numerical_forecasts.csv"
)

# Get final observations from December 2023
spy_history = spy_history.sort_values("Date")
cpi_history = cpi_history.sort_values("observation_date")

baseline_values = {
    "spy_price": float(spy_history["Close"].iloc[-1]),
    "inflation_rate": float(
        cpi_history["inflation_rate"].iloc[-1]
    ),
}

# Fix valid numeric outputs such as "0."
def parse_number(response):
    text = str(response).strip()
    if re.fullmatch(
        r"\$?\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*%?",
        text,
    ):
        return float(text.replace("$", "").replace("%", ""))
    return None


# Re-parse all original model responses consistently
results["prediction"] = results["raw_response"].apply(
    parse_number
)

results["absolute_error"] = (
    results["prediction"] - results["actual"]
).abs()

# Add naive baseline predictions
baseline_rows = []

for task, prediction in baseline_values.items():
    actual = float(
        results.loc[results["task"] == task, "actual"].iloc[0]
    )

    baseline_rows.append({
        "model": "Naive baseline",
        "task": task,
        "prediction": prediction,
        "actual": actual,
        "absolute_error": abs(prediction - actual),
    })

baseline_df = pd.DataFrame(baseline_rows)

comparison = pd.concat(
    [
        results[
            ["model", "task", "prediction",
             "actual", "absolute_error"]
        ],
        baseline_df,
    ],
    ignore_index=True,
)

# Flag physically invalid price predictions
comparison["valid_prediction"] = comparison["prediction"].notna()

comparison.loc[
    (comparison["task"] == "spy_price")
    & (comparison["prediction"] <= 0),
    "valid_prediction",
] = False

output_path = OUTPUT_DIR / "baseline_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\nForecast comparison:")
print(comparison.to_string(index=False))

print(f"\nSaved to: {output_path}")

