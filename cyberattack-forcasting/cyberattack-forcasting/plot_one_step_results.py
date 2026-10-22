
import pandas as pd
import matplotlib.pyplot as plt

# Load saved predictions
df = pd.read_csv(
    "data/one_step_predictions.csv",
    parse_dates=["Date"]
)

models = [
    ("Attention_LSTM", "Attention-LSTM"),
    ("Naive_Baseline", "Naive Baseline"),
    ("Random_Forest", "Random Forest"),
]

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(12, 10),
    sharex=True
)

for ax, (model_key, title) in zip(axes, models):
    subset = df[df["Model"] == model_key].sort_values("Date")

    ax.plot(
        subset["Date"],
        subset["Actual"],
        marker="o",
        label="Actual"
    )
    ax.plot(
        subset["Date"],
        subset["Predicted"],
        marker="x",
        linestyle="--",
        label="Predicted"
    )

    ax.set_title(title)
    ax.set_ylabel("Monthly attacks")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Month")

plt.suptitle(
    "Cyberattack Frequency Forecasting: Actual vs Predicted",
    fontsize=14
)
plt.tight_layout()

plt.savefig(
    "data/final_forecast_comparison.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

print("Saved: data/final_forecast_comparison.png")