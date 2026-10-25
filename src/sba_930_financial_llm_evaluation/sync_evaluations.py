
from pathlib import Path
import pandas as pd

opt_dir = Path("artifacts/optimization")

# Load corrected original evaluation
original = pd.read_csv(
    "artifacts/evaluation/model_evaluation.csv"
)

# Load completed optimization evaluations
path = opt_dir / "optimization_evaluated.csv"
df = pd.read_csv(path)

# Update original forecasting scores
for _, row in original.iterrows():
    mask = (
        (df["model"] == row["model"])
        & (df["category"] == "forecasting")
        & (df["task_name"] == row["task"])
    )

    if mask.sum() != 1:
        raise ValueError("Forecast evaluation row mismatch")

    df.loc[mask, "original_score"] = row["overall_score"]

# Correct the optimized market-volatility score
mask = (
    (df["model"] == "google/flan-t5-base")
    & (df["category"] == "risk_analysis")
    & (df["task_name"] == "market_volatility")
)

if mask.sum() != 1:
    raise ValueError("Market volatility row mismatch")

df.loc[mask, "optimized_score"] = (2 + 1 + 2) / 3

df.loc[mask, "evaluation_notes"] = (
    "The response vaguely identifies stock market risk "
    "but does not explain volatility or provide a "
    "practical mitigation strategy."
)

# Recalculate all improvement values
df["improvement"] = (
    df["optimized_score"] - df["original_score"]
)

# Save corrected detailed results
df.to_csv(path, index=False)

# Recalculate the summary
summary = df.groupby(["model", "category"])[
    ["original_score", "optimized_score", "improvement"]
].mean().round(2)

summary.to_csv(opt_dir / "optimization_summary.csv")

print("\nFinal Corrected Optimization Summary:")
print(summary.to_string())

print("\nAll evaluation corrections synchronized!")
