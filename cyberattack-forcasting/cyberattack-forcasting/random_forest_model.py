
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/monthly_attack_summary.csv"
WINDOW_SIZE = 8

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

values = df["Total_Attacks"].to_numpy(dtype=float)

# Build sliding windows: previous 8 months -> next month
X, y = [], []

for i in range(WINDOW_SIZE, len(values)):
    X.append(values[i - WINDOW_SIZE:i])
    y.append(values[i])

X = np.asarray(X)
y = np.asarray(y)

# Chronological split: 70% / 15% / 15%
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

def evaluate(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name} MAE:  {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")

print("Random Forest Forecasting Results")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

evaluate(y_val, val_predictions, "Validation")
evaluate(y_test, test_predictions, "Test")

print("\nFirst 5 test comparisons:")
print(pd.DataFrame({
    "Actual": y_test[:5],
    "Predicted": np.round(test_predictions[:5], 2)
}).to_string(index=False))