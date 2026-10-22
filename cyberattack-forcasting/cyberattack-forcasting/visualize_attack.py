
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/monthly_attack_summary.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date")

# Plot 1: Overall monthly frequency
plt.figure(figsize=(12, 5))
plt.plot(
    df["Date"],
    df["Total_Attacks"],
    label="Total recorded attack counts"
)
plt.title("Monthly Cyberattack Frequency")
plt.xlabel("Date")
plt.ylabel("Recorded Count")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("data/monthly_attack_trend.png", dpi=150)
plt.show()

# Plot 2: Most active categories
category_columns = [
    col for col in df.columns
    if col not in ("Date", "Total_Attacks")
]

top_categories = (
    df[category_columns].sum()
    .sort_values(ascending=False)
    .head(8)
    .index
)

plt.figure(figsize=(12, 5))

for category in top_categories:
    plt.plot(df["Date"], df[category], label=category)

plt.title("Monthly Trends of Top 8 Recorded Categories")
plt.xlabel("Date")
plt.ylabel("Recorded Count")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("data/attack_category_trends.png", dpi=150)
plt.show()

print("Top categories by cumulative recorded count:")
print(df[list(top_categories)].sum().sort_values(ascending=False))
print("\nCharts saved in the data folder.")