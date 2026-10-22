
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "data/monthly_attack_summary.csv"
OUTPUT_PATH = "data/attention_lstm_predictions.csv"

WINDOW_SIZE = 8
EPOCHS = 100
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 15

# --------------------------------------------------
# 2. Load monthly cyberattack data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

values = df["Total_Attacks"].to_numpy(
    dtype=np.float32
).reshape(-1, 1)

print("Dataset shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())

# --------------------------------------------------
# 3. Create sliding windows
# Previous 8 months -> predict next month
# --------------------------------------------------
X, y, target_dates = [], [], []

for i in range(WINDOW_SIZE, len(values)):
    X.append(values[i - WINDOW_SIZE:i])
    y.append(values[i])
    target_dates.append(df.loc[i, "Date"])

X = np.asarray(X, dtype=np.float32)  # (samples, 8, 1)
y = np.asarray(y, dtype=np.float32)  # (samples, 1)
target_dates = pd.to_datetime(target_dates)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

# --------------------------------------------------
# 4. Chronological train / validation / test split
# --------------------------------------------------
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_raw = X[:train_end]
y_train_raw = y[:train_end]

X_val_raw = X[train_end:val_end]
y_val_raw = y[train_end:val_end]

X_test_raw = X[val_end:]
y_test = y[val_end:].reshape(-1)

test_dates = target_dates[val_end:]

print("\nTraining examples:", len(X_train_raw))
print("Validation examples:", len(X_val_raw))
print("Testing examples:", len(X_test_raw))

# --------------------------------------------------
# 5. Scale using training-period data only
# --------------------------------------------------
scaler = MinMaxScaler()

# Fit only on observations available before the
# validation and test target periods.
scaler.fit(values[:train_end + WINDOW_SIZE])

X_train = scaler.transform(
    X_train_raw.reshape(-1, 1)
).reshape(X_train_raw.shape)

X_val = scaler.transform(
    X_val_raw.reshape(-1, 1)
).reshape(X_val_raw.shape)

X_test = scaler.transform(
    X_test_raw.reshape(-1, 1)
).reshape(X_test_raw.shape)

y_train = scaler.transform(y_train_raw).astype(np.float32)
y_val = scaler.transform(y_val_raw).astype(np.float32)

# Important: X arrays are already 3D.
# Shape = (samples, sequence_length, input_size)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

print("\nLSTM input shape:", tuple(X_train.shape))

# --------------------------------------------------
# 6. Attention-LSTM architecture
# --------------------------------------------------
class AttentionLSTM(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, 1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Output shape: (batch, sequence_length, hidden_size)
        sequence_output, _ = self.lstm(x)

        # Attention scores: (batch, sequence_length)
        scores = self.attention(
            sequence_output
        ).squeeze(-1)

        # Normalize scores across time steps
        weights = torch.softmax(scores, dim=1)

        # Weighted context vector: (batch, hidden_size)
        context = torch.sum(
            sequence_output * weights.unsqueeze(-1),
            dim=1
        )

        # Predict next month's scaled count
        return self.output(context)


model = AttentionLSTM(hidden_size=HIDDEN_SIZE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# 7. Training with validation and early stopping
# --------------------------------------------------
best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # X_train is already 3D: do NOT use unsqueeze()
    predictions = model(X_train)
    train_loss = criterion(predictions, y_train)

    train_loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss.item():.5f} | "
            f"Validation Loss: {val_loss:.5f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

training_time = time.time() - start_time

# Restore the best validation checkpoint
if best_state is not None:
    model.load_state_dict(best_state)

# --------------------------------------------------
# 8. Evaluate on the held-out test set
# --------------------------------------------------
model.eval()

with torch.no_grad():
    test_scaled = model(X_test).cpu().numpy().reshape(-1, 1)

test_predictions = scaler.inverse_transform(
    test_scaled
).reshape(-1)

mae = mean_absolute_error(y_test, test_predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

# Symmetric Mean Absolute Percentage Error
denominator = np.abs(y_test) + np.abs(test_predictions)

smape_terms = np.divide(
    2 * np.abs(y_test - test_predictions),
    denominator,
    out=np.zeros_like(denominator, dtype=float),
    where=denominator != 0
)

smape = np.mean(smape_terms) * 100

print("\n" + "=" * 45)
print("ATTENTION-LSTM TEST RESULTS")
print("=" * 45)
print("Test observations:", len(y_test))
print("Test period:", test_dates.min().date(), "to", test_dates.max().date())
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"SMAPE: {smape:.2f}%")
print(f"Training time: {training_time:.2f} seconds")

# --------------------------------------------------
# 9. Save predictions
# --------------------------------------------------
results = pd.DataFrame({
    "Date": test_dates,
    "Actual": y_test,
    "Attention_LSTM_Prediction": test_predictions
})

results.to_csv(OUTPUT_PATH, index=False)

print("\nFirst five test predictions:")
print(results.head().to_string(index=False))

print(f"\nPredictions saved to: {OUTPUT_PATH}")