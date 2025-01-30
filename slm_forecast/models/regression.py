from typing import Callable

import numpy as np
import pandas as pd

from main import ModelType
from slm_forecast.models.base import BaseModel

LAG_FEATURES = [1, 2, 3, 4, 12]
DIFF_LAG_FEATURES = [2, 4, 12]
ROLLING_WINDOWS = [2, 4, 8]


class RegressionModel(BaseModel):
    def __init__(
        self,
        create_base_model: Callable,
        use_fourier=False,
        use_cyclical=False,
    ):
        self.create_base_model = create_base_model
        self.use_fourier = use_fourier
        self.use_cyclical = use_cyclical

        self.models = {}
        self.historical_data = None

    def fit(self, historical_data):
        self.historical_data = historical_data.copy()
        self._create_time_index()

        for category in self.historical_data.columns[2:]:
            self._add_time_features(category)
            self._add_lag_features(category)
            self._add_rolling_features(category)
            self._add_lag_diff_features(category)

        if self.use_fourier:
            self._add_fourier_features()
        if self.use_cyclical:
            self._add_cyclical_features()

        for category in self.historical_data.columns[2:]:
            train_df = self.historical_data[self.historical_data["YEAR"] <= 2023]
            test_df = self.historical_data[self.historical_data["YEAR"] == 2024]

            feature_cols = self._feature_columns(category)
            train_df = train_df.dropna(subset=feature_cols + [category])
            test_df = test_df.dropna(subset=feature_cols)

            if train_df.empty:  # or test_df.empty:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df[category]
            model = self.create_base_model()
            model.fit(X_train, y_train)
            self.models[category] = model

    def predict(self, year, week):
        if not self.models:
            raise ValueError("Models are not trained.")
        row = self.historical_data[
            (self.historical_data["YEAR"] == year)
            & (self.historical_data["WEEKNUM"] == week)
        ]
        if row.empty:
            return pd.Series(np.nan, index=self.models.keys())

        predictions = {}
        for category, model in self.models.items():
            X = row[self._feature_columns(category)]
            predictions[category] = (
                model.predict(X.iloc[[0]])[0] if len(X) > 0 else np.nan
            )

        return pd.Series(predictions)

    # Exactly the same helpers as in LightGBMModel now:
    def _create_time_index(self):
        self.historical_data["time_index"] = np.arange(len(self.historical_data)) + 1

    def _add_time_features(self, category):
        self.historical_data[f"{category}_annual_avg"] = (
            self.historical_data[category]
            .rolling(window=52, min_periods=1)
            .mean()
            .shift(1)
        )
        self.historical_data[f"{category}_quarterly_avg"] = (
            self.historical_data[category]
            .rolling(window=13, min_periods=1)
            .mean()
            .shift(1)
        )

    def _add_lag_features(self, category):
        for lag in LAG_FEATURES:
            self.historical_data[f"{category}_lag_{lag}"] = self.historical_data[
                category
            ].shift(lag)

    def _add_lag_diff_features(self, category):
        lags = DIFF_LAG_FEATURES
        lag_1_col = f"{category}_lag_1"
        for lag in lags:
            lag_col = f"{category}_lag_{lag}"
            self.historical_data[f"{category}_lag_diff_1_{lag}"] = (
                self.historical_data[lag_1_col] - self.historical_data[lag_col]
            )

    def _add_rolling_features(self, category):
        for window in ROLLING_WINDOWS:
            self.historical_data[f"{category}_rolling_mean_{window}"] = (
                self.historical_data[category]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
            )

    def _add_fourier_features(self):
        week_of_year = self.historical_data["WEEKNUM"].values
        for k in [1, 2, 3]:
            self.historical_data[f"fourier_sin_{k}"] = np.sin(
                2 * k * np.pi * week_of_year / 52
            )
            self.historical_data[f"fourier_cos_{k}"] = np.cos(
                2 * k * np.pi * week_of_year / 52
            )

    def _add_cyclical_features(self):
        week_of_year = self.historical_data["WEEKNUM"].values
        self.historical_data["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
        self.historical_data["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)

    def _feature_columns(self, category):
        return [
            col
            for col in self.historical_data.columns
            if col != category
            and (
                col.startswith(f"{category}_lag")
                or col.startswith(f"{category}_rolling")
                or col.startswith(f"{category}_lag_diff")
                or col in ["time_index", "week_sin", "week_cos", "WEEKNUM", "YEAR"]
                or col.startswith("fourier")
            )
        ]
