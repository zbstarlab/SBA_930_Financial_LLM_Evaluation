
from pathlib import Path
import pandas as pd

INPUT = Path("artifacts/optimization/optimization_comparison.csv")
OUTPUT = Path("artifacts/optimization")

df = pd.read_csv(INPUT)
df["evaluation_notes"] = df["evaluation_notes"].astype("string")
df["optimized_score"] = df["optimized_score"].astype("float64")
df["improvement"] = df["improvement"].astype("float64")

for index, row in df.iterrows():
    print("\n" + "=" * 65)
    print(f"Model: {row['model']}")
    print(f"Category: {row['category']}")
    print(f"Task: {row['task_name']}")
    print(f"Original: {row['original_response']}")
    print(f"Optimized: {row['optimized_response']}")

    if row["category"] == "forecasting":
        criteria = [
            "relevance",
            "financial_accuracy",
            "reliability",
        ]
    else:
        criteria = [
            "risk_identification",
            "mitigation_quality",
            "clarity",
        ]

    scores = []

    for criterion in criteria:
        while True:
            value = input(f"{criterion} (1-5): ").strip()
            if value in ["1", "2", "3", "4", "5"]:
                scores.append(int(value))
                break
            print("Enter a number between 1 and 5.")

    optimized_score = sum(scores) / len(scores)

    df.loc[index, "optimized_score"] = optimized_score
    df.loc[index, "improvement"] = (
        optimized_score - row["original_score"]
    )
    df.loc[index, "evaluation_notes"] = input(
        "Reason for scores: "
    ).strip()

df.to_csv(
    OUTPUT / "optimization_evaluated.csv",
    index=False,
)

summary = df.groupby(["model", "category"])[
    ["original_score", "optimized_score", "improvement"]
].mean().round(2)

summary.to_csv(
    OUTPUT / "optimization_summary.csv"
)

print("\nOptimization Summary:")
print(summary.to_string())

print("\nOptimization evaluation completed!")

