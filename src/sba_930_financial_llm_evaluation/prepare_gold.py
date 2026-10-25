
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load historical gold ETF data
print("Preparing gold dataset...")

gold = pd.read_csv(
    RAW_DIR / "gold_historical.csv",
    parse_dates=["Date"],
)

# Remove duplicate dates and sort chronologically
gold = gold.drop_duplicates(subset=["Date"])
gold = gold.sort_values("Date")

# Check missing values
print("\nGold missing values:")
print(gold.isnull().sum())

# Remove rows with missing closing prices
gold = gold.dropna(subset=["Close"])

# Calculate daily percentage returns
gold["daily_return"] = gold["Close"].pct_change(
    fill_method=None
) * 100

# Split historical and evaluation periods
cutoff = pd.Timestamp("2024-01-01")

gold_history = gold[gold["Date"] < cutoff].copy()
gold_test = gold[gold["Date"] >= cutoff].copy()

# Save cleaned and split datasets
gold.to_csv(
    PROCESSED_DIR / "gold_clean.csv",
    index=False,
)

gold_history.to_csv(
    PROCESSED_DIR / "gold_history.csv",
    index=False,
)

gold_test.to_csv(
    PROCESSED_DIR / "gold_test.csv",
    index=False,
)

print(f"\nGold cleaned rows: {len(gold)}")
print(f"Gold historical rows: {len(gold_history)}")
print(f"Gold evaluation rows: {len(gold_test)}")

print("\nGold dataset prepared successfully!")
