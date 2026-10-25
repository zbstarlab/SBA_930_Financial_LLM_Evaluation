
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("artifacts/optimization")

input_path = OUTPUT_DIR / "optimization_evaluated.csv"
df = pd.read_csv(input_path)

# Find the specific evaluation that needs correction
mask = (
    (df["model"] == "google/flan-t5-base")
    & (df["category"] == "risk_analysis")
    & (df["task_name"] == "market_volatility")
)

if mask.sum() != 1:
    raise ValueError("Expected exactly one matching evaluation.")

# Correct scores: identification=2, mitigation=1, clarity=2
corrected_score = (2 + 1 + 2) / 3

df.loc[mask, "optimized_score"] = corrected_score

df.loc[mask, "improvement"] = (
    corrected_score - df.loc[mask, "original_score"]
)

df.loc[mask, "evaluation_notes"] = (
    "The response vaguely identifies stock market risk "
    "but does not specifically explain market volatility "
    "or provide a practical mitigation strategy."
)

# Save corrected detailed evaluation
df.to_csv(input_path, index=False)

# Recalculate summary
summary = df.groupby(["model", "category"])[
    ["original_score", "optimized_score", "improvement"]
].mean().round(2)

summary.to_csv(
    OUTPUT_DIR / "optimization_summary.csv"
)

print("\nCorrected Optimization Summary:")
print(summary.to_string())

print("\nCorrection completed successfully!")

