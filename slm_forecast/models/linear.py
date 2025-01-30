import pandas as pd
from sklearn.linear_model import LinearRegression

from slm_forecast.models.base import BaseModel
from slm_forecast.models.regression import RegressionModel


class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = RegressionModel(create_base_model=lambda: LinearRegression())

    def fit(self, historical_data: pd.DataFrame):
        self.model.fit(historical_data)

    def predict(self, year: int, week: int) -> pd.Series:
        return self.model.predict(year, week)
