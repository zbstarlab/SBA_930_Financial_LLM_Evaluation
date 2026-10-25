
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
]

RISK_SCENARIOS = {
    "market_volatility": (
        "Financial risk analysis: Stock prices have "
        "fallen by 15 percent in one month and market "
        "volatility has increased. Identify the main "
        "financial risk and suggest one mitigation strategy."
    ),

    "credit_risk": (
        "Financial risk analysis: A company has high "
        "debt, declining revenue, and difficulty making "
        "loan payments. Identify the main financial risk "
        "and suggest one mitigation strategy."
    ),

    "liquidity_risk": (
        "Financial risk analysis: A financial institution "
        "has insufficient cash to meet short-term "
        "withdrawal requests. Identify the main financial "
        "risk and suggest one mitigation strategy."
    ),
}

OUTPUT_DIR = Path("artifacts/risk_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

results = []

for model_name in MODELS:
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    for risk_type, prompt in RISK_SCENARIOS.items():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                no_repeat_ngram_size=3,
            )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        results.append({
            "model": model_name,
            "risk_type": risk_type,
            "prompt": prompt,
            "response": response,
        })

        print(f"\nRisk: {risk_type}")
        print(f"Response: {response}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "risk_responses.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print(f"Total responses: {len(results_df)}")

