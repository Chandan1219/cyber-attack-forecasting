
import pandas as pd

path = "data/research_tumbling_predictions.csv"
df = pd.read_csv(path, parse_dates=["Date"])

for name, group in df.groupby("Filter"):
    print(f"\n{'=' * 55}")
    print(f"FILTER: {name}")
    print("=" * 55)

    print(
        group[["Date", "Actual", "Predicted"]]
        .to_string(index=False, formatters={
            "Actual": "{:.0f}".format,
            "Predicted": "{:.2f}".format,
        })
    )