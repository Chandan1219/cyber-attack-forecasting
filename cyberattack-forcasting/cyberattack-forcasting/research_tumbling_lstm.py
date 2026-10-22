
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
SEED = 42
WINDOW_SIZE = 8
HIDDEN_SIZE = 50
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.01

DATA_PATH = Path("data/monthly_attack_summary.csv")
OUTPUT_PATH = Path("data/research_tumbling_results.csv")

np.random.seed(SEED)
torch.manual_seed(SEED)

# Paper's normalized filter weights for window size 8
FILTERS = {
    "No_Filter": np.ones(8, dtype=float) / 8,
    "STCF": np.array([
        0.0121, 0.0199, 0.0329, 0.0543,
        0.0895, 0.1476, 0.2432, 0.4008
    ]),
    "SMCF": np.array([
        0.4008, 0.2432, 0.1476, 0.0895,
        0.0543, 0.0329, 0.0199, 0.0121
    ]),
    "LCF": np.array([
        0.0603, 0.1134, 0.1528, 0.1735,
        0.1735, 0.1528, 0.1134, 0.0603
    ]),
}

# --------------------------------------------------
# 2. Attention-LSTM architecture
# --------------------------------------------------
class AttentionLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.attention_score = nn.Linear(HIDDEN_SIZE, 1)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)

        scores = self.attention_score(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        context = torch.sum(
            sequence * weights.unsqueeze(-1),
            dim=1
        )

        return self.output(context)


# --------------------------------------------------
# 3. Metrics
# --------------------------------------------------
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    denominator = np.abs(actual) + np.abs(predicted)
    terms = np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    return mae, rmse, terms.mean() * 100


# --------------------------------------------------
# 4. Train one model using history before a block
# --------------------------------------------------
def train_model(history, filter_weights):
    # Scale using history available at this forecast origin
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        np.asarray(history, dtype=float).reshape(-1, 1)
    ).flatten()

    X, y = [], []

    for i in range(WINDOW_SIZE, len(scaled)):
        window = scaled[i - WINDOW_SIZE:i].copy()

        # Apply the selected normalized filter to lag values
        window = window * filter_weights

        X.append(window.reshape(WINDOW_SIZE, 1))
        y.append(scaled[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    if len(X) < 10:
        raise ValueError(
            "Not enough historical sequences to train this model."
        )

    # Chronological validation split
    split = max(1, int(len(X) * 0.85))
    split = min(split, len(X) - 1)

    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32)
    X_val = torch.tensor(X[split:], dtype=torch.float32)
    y_val = torch.tensor(y[split:], dtype=torch.float32)

    model = AttentionLSTM()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    wait = 0

    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        train_prediction = model(X_train)
        train_loss = criterion(train_prediction, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(X_val)
            val_loss = criterion(val_prediction, y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    elapsed = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, elapsed


# --------------------------------------------------
# 5. Forecast one non-overlapping test block
# --------------------------------------------------
def forecast_block(model, scaler, history, block_size, weights):
    history_scaled = scaler.transform(
        np.asarray(history, dtype=float).reshape(-1, 1)
    ).flatten().tolist()

    predictions = []

    model.eval()

    for _ in range(block_size):
        window = np.array(
            history_scaled[-WINDOW_SIZE:],
            dtype=np.float32
        )

        window = window * weights

        x = torch.tensor(
            window.reshape(1, WINDOW_SIZE, 1),
            dtype=torch.float32
        )

        with torch.no_grad():
            prediction_scaled = model(x).item()

        prediction = scaler.inverse_transform(
            np.array([[prediction_scaled]])
        )[0, 0]

        predictions.append(prediction)

        # Recursive forecast: use the prediction for the next step
        history_scaled.append(prediction_scaled)

    return np.asarray(predictions)


# --------------------------------------------------
# 6. Load data and establish chronological split
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

dates = df["Date"].to_numpy()
values = df["Total_Attacks"].to_numpy(dtype=float)

n = len(values)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Validation period is kept separate from the final test period.
# Models use data available before each test block.
test_start = val_end

print("Dataset records:", n)
print("Training/validation history ends:", df.loc[test_start - 1, "Date"].date())
print("Test starts:", df.loc[test_start, "Date"].date())
print("Test ends:", df.loc[n - 1, "Date"].date())

# --------------------------------------------------
# 7. Tumbling-window experiment
# --------------------------------------------------
all_results = []
prediction_rows = []

for filter_name, weights in FILTERS.items():
    print(f"\nRunning filter: {filter_name}")

    history = list(values[:test_start])
    block_start = test_start
    predictions = []
    training_seconds = 0.0

    # Forecast blocks of 8, 8, and the remaining months.
    # Retrain at each non-overlapping block boundary.
    while block_start < n:
        block_size = min(WINDOW_SIZE, n - block_start)

        model, scaler, elapsed = train_model(
            history,
            weights
        )
        training_seconds += elapsed

        block_prediction = forecast_block(
            model,
            scaler,
            history,
            block_size,
            weights
        )

        predictions.extend(block_prediction)

        # Advance to the next block using observed data.
        # Within the current block, predictions were recursive.
        history.extend(
            values[block_start:block_start + block_size].tolist()
        )

        block_start += block_size

    actual = values[test_start:]
    predictions = np.asarray(predictions)

    mae, rmse, smape = calculate_metrics(actual, predictions)

    all_results.append({
        "Filter": filter_name,
        "Test_Observations": len(actual),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "SMAPE_Percent": round(smape, 2),
        "Total_Training_Seconds": round(training_seconds, 2),
    })

    for date, observed, predicted in zip(
        dates[test_start:], actual, predictions
    ):
        prediction_rows.append({
            "Filter": filter_name,
            "Date": pd.Timestamp(date),
            "Actual": observed,
            "Predicted": predicted,
        })

# --------------------------------------------------
# 8. Save and display results
# --------------------------------------------------
results_df = pd.DataFrame(all_results).sort_values("SMAPE_Percent")
predictions_df = pd.DataFrame(prediction_rows)

results_df.to_csv(OUTPUT_PATH, index=False)
predictions_df.to_csv(
    "data/research_tumbling_predictions.csv",
    index=False
)

print("\nRESEARCH-ALIGNED TUMBLING RESULTS")
print(results_df.to_string(index=False))

print("\nSaved:")
print(OUTPUT_PATH)
print("data/research_tumbling_predictions.csv")