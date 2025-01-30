import pandas as pd
from tqdm import tqdm

from slm_forecast.models.base import BaseModel


def fit_predict(
    model: BaseModel,
    df: pd.DataFrame,
    start_year=2024,
    start_week=1,
    n_weeks=52,
) -> pd.DataFrame:
    train_df = df[
        (100 * df["YEAR"] + df["WEEKNUM"]) <= 100 * start_year + start_week
    ].copy()
    predictions = []
    for i in tqdm(range(n_weeks)):
        current_year = start_year + (start_week + i - 1) // 52
        current_week = (start_week + i - 1) % 52 + 1
        model.fit(train_df)
        pred_series = model.predict(current_year, current_week)
        pred_row = pd.DataFrame([pred_series])
        pred_row["YEAR"] = current_year
        pred_row["WEEKNUM"] = current_week
        next_year = current_year + 1 if current_week == 52 else current_year
        next_week = 1 if current_week == 52 else current_week + 1
        actual_row = df[(df["YEAR"] == next_year) & (df["WEEKNUM"] == next_week)]
        predictions.append(pred_row)
        if not actual_row.empty:
            train_df = pd.concat([train_df, actual_row], ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    return predictions_df
