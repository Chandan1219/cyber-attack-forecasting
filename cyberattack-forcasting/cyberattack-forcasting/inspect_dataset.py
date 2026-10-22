
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/T-HMGDN-F-0711-0322.csv")
OUTPUT_PATH = Path("data/monthly_attack_summary.csv")

df = pd.read_csv(DATA_PATH)

# Parse the monthly dates.
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%b-%y",
    errors="raise"
)

df = df.sort_values("Date").reset_index(drop=True)

# Identify country-level attack columns.
# Example: DDoS-US, Ransomware-IN
excluded_prefixes = ("Mentions-",)
attack_columns = [
    col for col in df.columns
    if "-" in col
    and not col.startswith(excluded_prefixes)
]

# Aggregate counts across countries for each attack type.
summary = pd.DataFrame({"Date": df["Date"]})

for col in attack_columns:
    attack_type = col.rsplit("-", 1)[0]

    if attack_type not in summary.columns:
        summary[attack_type] = 0

    summary[attack_type] += pd.to_numeric(
        df[col], errors="raise"
    )

# Total across the identified attack categories.
category_columns = [
    col for col in summary.columns if col != "Date"
]
summary["Total_Attacks"] = summary[category_columns].sum(axis=1)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_PATH, index=False)

print("Dataset inspection completed.")
print(f"Original shape: {df.shape}")
print(f"Date range: {df['Date'].min():%b %Y} to "
      f"{df['Date'].max():%b %Y}")
print(f"Monthly records: {len(summary)}")
print(f"Attack categories: {category_columns}")
print("\nFirst 5 monthly records:")
print(summary.head().to_string(index=False))
print(f"\nSaved summary to: {OUTPUT_PATH}")