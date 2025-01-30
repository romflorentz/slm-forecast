import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(df_hist: pd.DataFrame, df_pred: pd.DataFrame):
    EXCLUDED_CATEGORIES = ["APP RUNNING", "APP SKI", "APP OTHER", "APP OUTDOOR"]
    df_hist.drop(columns=EXCLUDED_CATEGORIES, inplace=True)
    df_pred.drop(columns=EXCLUDED_CATEGORIES, inplace=True)
    df_merged = df_hist.merge(
        df_pred, on=["YEAR", "WEEKNUM"], suffixes=("_true", "_pred")
    )
    common_columns = [
        col for col in df_hist.columns if col not in ["YEAR", "WEEKNUM", "DATE"]
    ]
    if not common_columns:
        print("No common columns to evaluate.")
        return

    metrics_df = pd.DataFrame(
        index=pd.Index(common_columns, name="Category"),
        columns=["MAE", "MSE", "RMSE", "R²", "MAPE", "sMAPE", "Forecast Bias"],
    )

    for category in common_columns:
        true_col = f"{category}_true"
        pred_col = f"{category}_pred"

        if true_col not in df_merged.columns or pred_col not in df_merged.columns:
            print(f"Skipping {category}: Columns not found in merged data.")
            continue

        y_true = df_merged[true_col]
        y_pred = df_merged[pred_col]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = 100 * np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
        )
        bias = np.mean(y_true - y_pred)
        metrics_df.loc[category] = [
            f"{x:.2f}" for x in [mae, mse, rmse, r2, mape, smape, bias]
        ]

    return metrics_df
