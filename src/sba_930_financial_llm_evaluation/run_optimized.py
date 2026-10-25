
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .optimized_prompts import (
    OPTIMIZED_FORECAST_PROMPTS,
    OPTIMIZED_RISK_PROMPTS,
)

MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
]

OUTPUT_DIR = Path("artifacts/optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

results = []

experiments = {
    "forecasting": OPTIMIZED_FORECAST_PROMPTS,
    "risk_analysis": OPTIMIZED_RISK_PROMPTS,
}

for model_name in MODELS:
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    for category, prompts in experiments.items():
        for task_name, prompt in prompts.items():
            print(f"\nCategory: {category}")
            print(f"Task: {task_name}")

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
                )

            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            results.append({
                "model": model_name,
                "category": category,
                "task": task_name,
                "prompt": prompt,
                "response": response,
            })

            print(f"Response: {response}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "optimized_responses.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print(f"Total responses: {len(results_df)}")
