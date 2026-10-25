from pathlib import Path
import pandas as pd

INPUT_PATH = Path("artifacts/forecasts/model_responses.csv")
OUTPUT_DIR = Path("artifacts/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

criteria = {
    "relevance": (
        "1 = unrelated, 3 = partially addresses the task, "
        "5 = directly and thoroughly addresses the task"
    ),
    "financial_accuracy": (
        "1 = incorrect or misleading, "
        "3 = partially correct or insufficiently supported, "
        "5 = financially sound with appropriate reasoning"
    ),
    "reliability": (
        "1 = incoherent or highly unreliable, "
        "3 = understandable but vague or incomplete, "
        "5 = coherent, specific, and appropriately qualified"
    ),
}

print("\nEvaluation criteria:")
for name, description in criteria.items():
    print(f"{name}: {description}")

scores = []

for _, row in df.iterrows():
    print("\n" + "=" * 65)
    print(f"Model: {row['model']}")
    print(f"Task: {row['task']}")
    print(f"Prompt: {row['prompt']}")
    print(f"Response: {row['response']}")

    result = {
        "model": row["model"],
        "task": row["task"],
        "response": row["response"],
    }

    for criterion in criteria:
        while True:
            value = input(f"{criterion} (1-5): ").strip()

            if value in {"1", "2", "3", "4", "5"}:
                result[criterion] = int(value)
                break

            print("Enter a whole number between 1 and 5.")

    result["notes"] = input(
        "Brief reason for your scores: "
    ).strip()

    scores.append(result)

results = pd.DataFrame(scores)

results["overall_score"] = results[
    ["relevance", "financial_accuracy", "reliability"]
].mean(axis=1)

output_path = OUTPUT_DIR / "model_evaluation.csv"
results.to_csv(output_path, index=False)

summary = results.groupby("model")[
    [
        "relevance",
        "financial_accuracy",
        "reliability",
        "overall_score",
    ]
].mean().round(2)

summary_path = OUTPUT_DIR / "evaluation_summary.csv"
summary.to_csv(summary_path)

print("\nModel evaluation summary:")
print(summary)

print(f"\nDetailed results saved to: {output_path}")
print(f"Summary saved to: {summary_path}")
