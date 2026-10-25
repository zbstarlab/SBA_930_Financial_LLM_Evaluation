from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

# Load prepared datasets
spy = pd.read_csv(
    DATA_DIR / "spy_clean.csv",
    parse_dates=["Date"],
)

cpi = pd.read_csv(
    DATA_DIR / "cpi_clean.csv",
    parse_dates=["observation_date"],
)

# Divide into historical and evaluation periods
cutoff = pd.Timestamp("2024-01-01")

spy_history = spy[spy["Date"] < cutoff].copy()
spy_test = spy[spy["Date"] >= cutoff].copy()

cpi_history = cpi[
    cpi["observation_date"] < cutoff
].copy()

cpi_test = cpi[
    cpi["observation_date"] >= cutoff
].copy()

# Save the datasets
datasets = {
    "spy_history.csv": spy_history,
    "spy_test.csv": spy_test,
    "cpi_history.csv": cpi_history,
    "cpi_test.csv": cpi_test,
}

for filename, dataset in datasets.items():
    output_path = DATA_DIR / filename
    dataset.to_csv(output_path, index=False)
    print(f"Saved {filename}: {len(dataset)} rows")

print("\nHistorical forecasting datasets prepared!")
