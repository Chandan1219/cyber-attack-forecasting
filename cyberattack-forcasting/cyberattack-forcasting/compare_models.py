
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/monthly_attack_summary.csv"
WINDOW_SIZE = 8

# Load and sort monthly data
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

values = df["Total_Attacks"].to_numpy(dtype=float)

# Create 8-month input windows to predict the following month
X, y, target_dates = [], [], []

for i in range(WINDOW_SIZE, len(values)):
    X.append(values[i - WINDOW_SIZE:i])
    y.append(values[i])
    target_dates.append(df.loc[i, "Date"])

X = np.asarray(X)
y = np.asarray(y)
target_dates = pd.to_datetime(target_dates)

# Chronological split
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Model 1: Naive forecast - use the previous month's count
naive_predictions = X_test[:, -1]

# Model 2: Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)


def smape(actual, predicted):
    denominator = np.abs(actual) + np.abs(predicted)
    terms = np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0
    )
    return np.mean(terms) * 100


def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    score = smape(actual, predicted)

    return {
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "SMAPE (%)": round(score, 2)
    }


results = pd.DataFrame([
    evaluate("Naive Baseline", y_test, naive_predictions),
    evaluate("Random Forest", y_test, rf_predictions)
])

print("COMMON TEST PERIOD")
#print(target_dates.iloc[val_end].date(), "to", target_dates.iloc[-1].date())
print(target_dates[val_end].date(), "to", target_dates[-1].date())
print("Test observations:", len(y_test))

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

# Save results and individual predictions
results.to_csv("data/model_comparison.csv", index=False)

predictions_df = pd.DataFrame({
    "Date": target_dates[val_end:],
    "Actual": y_test,
    "Naive_Prediction": naive_predictions,
    "Random_Forest_Prediction": rf_predictions
})
predictions_df.to_csv("data/test_predictions.csv", index=False)

print("\nSaved:")
print("data/model_comparison.csv")
print("data/test_predictions.csv")