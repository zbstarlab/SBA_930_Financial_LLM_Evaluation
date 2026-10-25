# SBA 930 – Evaluating LLMs for Financial Forecasting and Risk Analysis

## Project Overview

This project evaluates the performance of two Hugging Face language models, Google FLAN-T5 Small and FLAN-T5 Base, in financial forecasting and risk analysis. Historical financial data was used to test the models' ability to generate forecasts, identify financial risks, and recommend mitigation strategies.

## Models Used

- Google FLAN-T5 Small (~80M parameters)
- Google FLAN-T5 Base (~250M parameters)

Both models were executed locally using Python, Hugging Face Transformers, and PyTorch.

## Datasets

- **SPY ETF:** Historical stock market prices from 2020–2024.
- **GLD ETF:** Historical gold ETF prices used as a commodity-price proxy.
- **US CPI:** Monthly Consumer Price Index data from FRED.

## Key Findings

- Neither model outperformed the naive baseline in the numerical forecasting experiments.
- FLAN-T5 Base performed better at identifying financial risks.
- FLAN-T5 Small achieved a higher score in the original qualitative financial analysis evaluation.
- Prompt optimization produced mixed results and did not consistently improve response quality.
- Both models showed limitations in financial reasoning and practical risk mitigation.

## Evaluation Methods

The models were evaluated using numerical forecast absolute error, manually scored response quality, risk identification and mitigation assessments, and original-versus-optimized prompt comparisons.

## Final Submission Report

**[View Final SBA 930 Submission (PDF)](docs/SBA_930_Per_Scholas_Submission.pdf)**

**[View Detailed Technical Report](docs/SBA_930_Final_Report.md)**

The final submission follows the Per Scholas report template and includes the project overview, implementation steps, evaluation results, screenshots, conclusion, and appendix.

## Technologies

Python 3.12, pandas, NumPy, Hugging Face Transformers, PyTorch, yfinance, and uv.

## Project Structure

- `src/` – Python implementation
- `data/` – Raw and processed financial datasets
- `artifacts/` – Forecasts and evaluation results
- `docs/` – Final submission report
- `reports/` – Working report and supporting analysis

## Reproducing the Experiments

Install project dependencies:

```bash
uv sync
```

Run the financial forecasting, risk analysis, and evaluation scripts from the project root using:

```bash
uv run python -m sba_930_financial_llm_evaluation.<module_name>
```

Replace `<module_name>` with the Python module you want to execute. The full methodology and results are documented in the final report.