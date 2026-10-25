import pandas as pd

from pathlib import Path

import yfinance as yf

# Define the directory where raw datasets will be stored
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download historical SPY prices
print("Downloading historical SPY stock data...")

spy_data = yf.download(
    "SPY",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False,
)

# Make sure the download returned data
if spy_data.empty:
    raise RuntimeError("SPY download failed or returned no data.")

# Flatten column names if yfinance returns a MultiIndex
if isinstance(spy_data.columns, __import__("pandas").MultiIndex):
    spy_data.columns = spy_data.columns.get_level_values(0)

# Save to CSV
output_path = DATA_DIR / "spy_historical.csv"
spy_data.to_csv(output_path)

print(f"Dataset saved to: {output_path}")
print(f"Total rows: {len(spy_data)}")
print(spy_data.head())



# Download CPI data from FRED
print("\nDownloading US CPI inflation data...")

cpi_url = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=CPIAUCSL"
    "&cosd=2020-01-01"
    "&coed=2024-12-31"
)

cpi_data = pd.read_csv(cpi_url)

# Convert dates to datetime
cpi_data["observation_date"] = pd.to_datetime(
    cpi_data["observation_date"]
)

# Calculate year-over-year inflation rate
cpi_data["inflation_rate"] = (
    cpi_data["CPIAUCSL"].pct_change(periods=12, fill_method=None) * 100
)

# Save the dataset
cpi_output_path = DATA_DIR / "cpi_inflation.csv"
cpi_data.to_csv(cpi_output_path, index=False)

print(f"Dataset saved to: {cpi_output_path}")
print(f"Total rows: {len(cpi_data)}")
print(cpi_data.head(15))


# Download historical gold ETF data
print("\nDownloading historical gold ETF data...")

gold_data = yf.download(
    "GLD",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False,
)

if gold_data.empty:
    raise RuntimeError("GLD download failed.")

# Flatten MultiIndex columns if present
if isinstance(gold_data.columns, pd.MultiIndex):
    gold_data.columns = gold_data.columns.get_level_values(0)

output_path = DATA_DIR / "gold_historical.csv"
gold_data.to_csv(output_path)

print(f"Dataset saved to: {output_path}")
print(f"Total rows: {len(gold_data)}")
print(gold_data.head())
