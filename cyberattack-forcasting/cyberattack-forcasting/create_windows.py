
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/monthly_attack_summary.csv")

WINDOW_SIZE = 8

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

values = df["Total_Attacks"].to_numpy(dtype=np.float32)
dates = df["Date"].to_numpy()

X = []
y = []
target_dates = []

for i in range(WINDOW_SIZE, len(values)):
    X.append(values[i - WINDOW_SIZE:i])
    y.append(values[i])
    target_dates.append(dates[i])

X = np.array(X)
y = np.array(y)
target_dates = np.array(target_dates)

print("Window size:", WINDOW_SIZE)
print("Input shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst training example:")
print("Previous 8 months:", X[0])
print("Target month:", pd.Timestamp(target_dates[0]))
print("Target value:", y[0])

print("\nTotal forecasting examples:", len(X))