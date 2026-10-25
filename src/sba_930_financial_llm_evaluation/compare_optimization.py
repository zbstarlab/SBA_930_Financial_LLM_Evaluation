
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("artifacts/optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original evaluations
forecast_original = pd.read_csv(
    "artifacts/evaluation/model_evaluation.csv"
)

risk_original = pd.read_csv(
    "artifacts/risk_analysis/risk_evaluation.csv"
)

# Load optimized model responses
optimized = pd.read_csv(
    OUTPUT_DIR / "optimized_responses.csv"
)

# Prepare original forecasting evaluations
forecast_original = forecast_original.rename(
    columns={"task": "task_name"}
)
forecast_original["category"] = "forecasting"

# Prepare original risk evaluations
risk_original = risk_original.rename(
    columns={
        "risk_type": "task_name",
        "risk_identification": "relevance",
        "mitigation_quality": "financial_accuracy",
    }
)
risk_original["category"] = "risk_analysis"

# Keep original scores and responses
score_columns = [
    "model",
    "category",
    "task_name",
    "response",
    "overall_score",
]

original = pd.concat(
    [
        forecast_original[score_columns],
        risk_original[score_columns],
    ],
    ignore_index=True,
)

original = original.rename(
    columns={
        "response": "original_response",
        "overall_score": "original_score",
    }
)

# Prepare optimized results
optimized = optimized.rename(
    columns={
        "task": "task_name",
        "response": "optimized_response",
    }
)

# Combine original and optimized outputs
comparison = original.merge(
    optimized[
        [
            "model",
            "category",
            "task_name",
            "optimized_response",
        ]
    ],
    on=["model", "category", "task_name"],
    how="inner",
    validate="one_to_one",
)

# Leave new evaluation fields empty
comparison["optimized_score"] = pd.NA
comparison["improvement"] = pd.NA
comparison["evaluation_notes"] = ""

output_path = OUTPUT_DIR / "optimization_comparison.csv"

comparison.to_csv(output_path, index=False)

print("\nOptimization comparison prepared!")
print(f"Total comparison rows: {len(comparison)}")
print(f"Saved to: {output_path}")

print(
    comparison[
        ["model", "category", "task_name", "original_score"]
    ].to_string(index=False)
)
