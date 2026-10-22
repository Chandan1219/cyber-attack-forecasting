
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/monthly_attack_summary.csv")
OUTPUT_PATH = Path("data/forecasting_dataset.csv")

# Load monthly attack data
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Select the forecasting target
target = df[["Date", "Total_Attacks"]].copy()

# Create a column containing the previous month's count
target["Previous_Month_Attacks"] = target["Total_Attacks"].shift(1)

# Remove the first row because it has no previous month
target = target.dropna().reset_index(drop=True)

# Chronological split: 70% train, 15% validation, 15% test
n = len(target)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = target.iloc[:train_end]
validation = target.iloc[train_end:val_end]
test = target.iloc[val_end:]

# Save prepared data
target.to_csv(OUTPUT_PATH, index=False)

print("Prepared dataset shape:", target.shape)
print("Training records:", len(train))
print("Validation records:", len(validation))
print("Testing records:", len(test))
print("\nFirst five prepared rows:")
print(target.head())
print("\nDate ranges:")
print("Train:", train["Date"].min(), "to", train["Date"].max())
print("Validation:", validation["Date"].min(), "to", validation["Date"].max())
print("Test:", test["Date"].min(), "to", test["Date"].max())
print("\nSaved to:", OUTPUT_PATH)