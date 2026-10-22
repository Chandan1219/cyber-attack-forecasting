
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
SEED = 42
WINDOW_SIZE = 8
HIDDEN_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = Path("data/monthly_attack_summary.csv")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load monthly data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

dates = df["Date"].to_numpy()
values = df["Total_Attacks"].to_numpy(dtype=float)

n = len(values)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Keep the final 15% for testing.
test_start = val_end

# Build one-step examples. Each input contains 8 actual months.
X, y, target_indices = [], [], []

for i in range(WINDOW_SIZE, n):
    X.append(values[i - WINDOW_SIZE:i])
    y.append(values[i])
    target_indices.append(i)

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)
target_indices = np.asarray(target_indices)

# Map the chronological boundary to the window examples.
train_mask = target_indices < train_end
val_mask = (target_indices >= train_end) & (target_indices < val_end)
test_mask = target_indices >= test_start

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_indices = target_indices[test_mask]
test_dates = dates[test_indices]

print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))
print("Test examples:", len(X_test))
print("Test period:", pd.Timestamp(test_dates[0]).date(),
      "to", pd.Timestamp(test_dates[-1]).date())

# --------------------------------------------------
# 3. Metrics
# --------------------------------------------------
def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    denominator = np.abs(actual) + np.abs(predicted)
    terms = np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    smape = terms.mean() * 100

    return mae, rmse, smape


# --------------------------------------------------
# 4. Naive baseline
# Prediction = previous actual month
# --------------------------------------------------
naive_predictions = X_test[:, -1]

# --------------------------------------------------
# 5. Random Forest
# Fit on training examples only.
# --------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=SEED,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# --------------------------------------------------
# 6. Attention-LSTM
# --------------------------------------------------
class AttentionLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.attention = nn.Linear(HIDDEN_SIZE, 1)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        scores = self.attention(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(
            sequence * weights.unsqueeze(-1), dim=1
        )
        return self.output(context)


# Fit the scaler using training-era observations only.
scaler = MinMaxScaler()
scaler.fit(values[:train_end].reshape(-1, 1))

X_train_scaled = scaler.transform(
    X_train.reshape(-1, 1)
).reshape(-1, WINDOW_SIZE, 1)

X_val_scaled = scaler.transform(
    X_val.reshape(-1, 1)
).reshape(-1, WINDOW_SIZE, 1)

X_test_scaled = scaler.transform(
    X_test.reshape(-1, 1)
).reshape(-1, WINDOW_SIZE, 1)

y_train_scaled = scaler.transform(
    y_train.reshape(-1, 1)
)
y_val_scaled = scaler.transform(
    y_val.reshape(-1, 1)
)

Xtr = torch.tensor(X_train_scaled, dtype=torch.float32)
Ytr = torch.tensor(y_train_scaled, dtype=torch.float32)
Xv = torch.tensor(X_val_scaled, dtype=torch.float32)
Yv = torch.tensor(y_val_scaled, dtype=torch.float32)
Xte = torch.tensor(X_test_scaled, dtype=torch.float32)

model = AttentionLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_state = None
wait = 0

start_time = time.time()

for epoch in range(MAX_EPOCHS):
    model.train()
    optimizer.zero_grad()

    train_loss = criterion(model(Xtr), Ytr)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(Xv), Yv).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if wait >= PATIENCE:
        break

training_seconds = time.time() - start_time

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    lstm_scaled = model(Xte).numpy()

lstm_predictions = scaler.inverse_transform(lstm_scaled).flatten()

# --------------------------------------------------
# 7. Compare on identical test dates
# --------------------------------------------------
prediction_map = {
    "Naive_Baseline": naive_predictions,
    "Random_Forest": rf_predictions,
    "Attention_LSTM": lstm_predictions,
}

results = []

for name, predictions in prediction_map.items():
    mae, rmse, smape = metrics(y_test, predictions)

    results.append({
        "Model": name,
        "Test_Observations": len(y_test),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "SMAPE_Percent": round(smape, 2),
    })

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv(OUTPUT_DIR / "one_step_model_comparison.csv", index=False)

prediction_rows = []

for name, predictions in prediction_map.items():
    for date, actual, predicted in zip(
        test_dates, y_test, predictions
    ):
        prediction_rows.append({
            "Model": name,
            "Date": pd.Timestamp(date),
            "Actual": actual,
            "Predicted": predicted,
        })

predictions_df = pd.DataFrame(prediction_rows)
predictions_df.to_csv(
    OUTPUT_DIR / "one_step_predictions.csv",
    index=False,
)

print("\nONE-STEP-AHEAD MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nLSTM training time:", round(training_seconds, 2), "seconds")
print("\nSaved:")
print("data/one_step_model_comparison.csv")
print("data/one_step_predictions.csv")