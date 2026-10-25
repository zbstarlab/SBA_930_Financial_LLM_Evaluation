
import pandas as pd
from pathlib import Path

evaluation_dir = Path("artifacts/evaluation")
file_path = evaluation_dir / "model_evaluation.csv"

df = pd.read_csv(file_path)

# Locate the incorrect evaluation
mask = (
    (df["model"] == "google/flan-t5-small")
    & (df["task"] == "commodity_prices")
)

if mask.sum() != 1:
    raise ValueError("Expected exactly one matching row")

# Correct reliability score
df.loc[mask, "reliability"] = 1

# Recalculate the overall scores
df["overall_score"] = df[
    ["relevance", "financial_accuracy", "reliability"]
].mean(axis=1)

# Save corrected evaluation
df.to_csv(file_path, index=False)

# Recalculate model summary
summary = df.groupby("model")[
    ["relevance", "financial_accuracy", "reliability", "overall_score"]
].mean().round(2)

summary.to_csv(
    evaluation_dir / "evaluation_summary.csv"
)

print("\nCorrected Original Evaluation Summary:")
print(summary.to_string())
print("\nCorrection completed successfully!")

