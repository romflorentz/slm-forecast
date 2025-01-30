import numpy as np
import pandas as pd

from slm_forecast.models.base import BaseModel, ModelType


class RatioBasedModel(BaseModel):
    name = ModelType.RATIO_BASED

    def __init__(self):
        self.historical_data = None

    def fit(self, historical_data):
        self.historical_data = historical_data

    def predict(self, year, week):
        if self.historical_data is None:
            raise ValueError("Model not fitted.")

        same_week_last_year = self.historical_data[
            (self.historical_data["YEAR"] == year - 1)
            & (self.historical_data["WEEKNUM"] == week)
        ]

        if same_week_last_year.empty:
            return pd.Series(np.nan, index=self.historical_data.columns[2:])

        prev_week_this_year = self.historical_data[
            (self.historical_data["YEAR"] == year)
            & (self.historical_data["WEEKNUM"] == week - 1)
        ]
        prev_week_last_year = self.historical_data[
            (self.historical_data["YEAR"] == year - 1)
            & (self.historical_data["WEEKNUM"] == week - 1)
        ]

        if prev_week_this_year.empty or prev_week_last_year.empty:
            return pd.Series(
                same_week_last_year.iloc[0, 2:].values,
                index=same_week_last_year.columns[2:],
            )

        prev_week_this_year_sales = np.expm1(prev_week_this_year.iloc[:, 2:].values)
        prev_week_last_year_sales = np.expm1(prev_week_last_year.iloc[:, 2:].values)
        same_week_last_year_sales = np.expm1(same_week_last_year.iloc[:, 2:].values)
        ratio = prev_week_this_year_sales / prev_week_last_year_sales
        prediction_sales = same_week_last_year_sales * ratio
        prediction = np.log1p(prediction_sales)

        return pd.Series(prediction.flatten(), index=same_week_last_year.columns[2:])
