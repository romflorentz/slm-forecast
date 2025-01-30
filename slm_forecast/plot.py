import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates as mdates


def plot(
    df_hist: pd.DataFrame, df_pred: pd.DataFrame, model_type: str, output_path: str
):
    for df in [df_hist, df_pred]:
        df["DATE"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" + df["WEEKNUM"].astype(str) + "-1",
            format="%Y-%W-%w",
        )

    # Filter df_pred to only include dates within the range of df_hist
    min_date = df_hist["DATE"].min()
    max_date = df_hist["DATE"].max()
    df_pred = df_pred[(df_pred["DATE"] >= min_date) & (df_pred["DATE"] <= max_date)]

    common_columns = set(df_hist.columns).intersection(set(df_pred.columns))
    common_columns = [
        col for col in common_columns if col not in ["YEAR", "WEEKNUM", "DATE"]
    ]

    if not common_columns:
        print("No common columns to plot.")
        return

    n_cols = 2
    n_rows = (len(common_columns) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, category in enumerate(sorted(common_columns)):
        ax = axes[idx]
        ax.plot(
            df_hist["DATE"],
            df_hist[category],
            label="Actual Sales",
            color="blue",
            marker="o",
            markersize=4,
            linewidth=1,
        )
        ax.plot(
            df_pred["DATE"],
            df_pred[category],
            label="Predicted Sales",
            color="red",
            linestyle="--",
            marker="x",
            markersize=5,
            linewidth=1,
        )
        ax.set_title(f"{category} Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales Volume")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45)

    for j in range(len(common_columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout so title is not overlapped
    plt.suptitle(
        f"Sales Forecast using {model_type.name} model",
        fontsize=24,
        fontweight="bold",
    )
    plt.savefig(output_path)
