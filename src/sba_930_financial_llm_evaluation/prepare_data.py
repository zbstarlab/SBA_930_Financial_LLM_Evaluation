from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load SPY data
print("Preparing SPY dataset...")

spy = pd.read_csv(
    RAW_DIR / "spy_historical.csv",
    parse_dates=["Date"]
)

# Remove duplicates and sort by date
spy = spy.drop_duplicates(subset=["Date"])
spy = spy.sort_values("Date")

# Check for missing values
print("SPY missing values:")
print(spy.isnull().sum())

# Remove rows with missing essential price data
spy = spy.dropna(subset=["Close"])

# Calculate daily percentage returns
spy["daily_return"] = spy["Close"].pct_change(
    fill_method=None
) * 100

# Save cleaned dataset
spy.to_csv(
    PROCESSED_DIR / "spy_clean.csv",
    index=False
)

print(f"SPY cleaned rows: {len(spy)}")


# Step 2: Load CPI data
print("\nPreparing CPI dataset...")

cpi = pd.read_csv(
    RAW_DIR / "cpi_inflation.csv",
    parse_dates=["observation_date"]
)

# Remove duplicates and sort by date
cpi = cpi.drop_duplicates(subset=["observation_date"])
cpi = cpi.sort_values("observation_date")

# Check missing values
print("CPI missing values:")
print(cpi.isnull().sum())

# Keep all CPI observations, including the first
# 12 rows where annual inflation cannot be calculated.

cpi.to_csv(
    PROCESSED_DIR / "cpi_clean.csv",
    index=False
)

print(f"CPI cleaned rows: {len(cpi)}")

print("\nFinancial datasets prepared successfully!")
