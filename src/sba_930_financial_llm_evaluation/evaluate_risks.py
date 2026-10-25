
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("artifacts/risk_analysis/risk_responses.csv")
OUTPUT_DIR = Path("artifacts/risk_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

scores = []

for _, row in df.iterrows():
    print("\n" + "=" * 60)
    print(f"Model: {row['model']}")
    print(f"Risk: {row['risk_type']}")
    print(f"Response: {row['response']}")

    result = {
        "model": row["model"],
        "risk_type": row["risk_type"],
        "response": row["response"],
    }

    for criterion in [
        "risk_identification",
        "mitigation_quality",
        "clarity",
    ]:
        while True:
            value = input(f"{criterion} (1-5): ").strip()

            if value in {"1", "2", "3", "4", "5"}:
                result[criterion] = int(value)
                break

            print("Please enter a number between 1 and 5.")

    result["notes"] = input(
        "Reason for scores: "
    ).strip()

    scores.append(result)

results = pd.DataFrame(scores)

results["overall_score"] = results[
    ["risk_identification", "mitigation_quality", "clarity"]
].mean(axis=1)

results.to_csv(
    OUTPUT_DIR / "risk_evaluation.csv",
    index=False,
)

summary = results.groupby("model")[
    [
        "risk_identification",
        "mitigation_quality",
        "clarity",
        "overall_score",
    ]
].mean().round(2)

summary.to_csv(
    OUTPUT_DIR / "risk_evaluation_summary.csv"
)

print("\nRisk evaluation summary:")
print(summary)

print("\nRisk evaluation completed successfully!")

