
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/monthly_attack_summary.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

values = df["Total_Attacks"].to_numpy(dtype=float)

# Predict the next month using the current month's count.
actual = values[1:]
predicted = values[:-1]

# Preserve chronological order: 70% train, 15% validation, 15% test
n = len(actual)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

y_test = actual[val_end:]
pred_test = predicted[val_end:]

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

print("Baseline Model: Previous Month Prediction")
print("Test records:", len(y_test))
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("\nFirst 5 test comparisons:")
print(pd.DataFrame({
    "Actual": y_test[:5],
    "Predicted": pred_test[:5]
}).to_string(index=False))