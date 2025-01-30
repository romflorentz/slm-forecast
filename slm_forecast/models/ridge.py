import pandas as pd
from sklearn.linear_model import Ridge

from slm_forecast.models.base import BaseModel
from slm_forecast.models.regression import RegressionModel


class RidgeRegressionModel(BaseModel):
    def __init__(self, alpha=1.0):
        self.model = RegressionModel(create_base_model=lambda: Ridge(alpha=alpha))

    def fit(self, historical_data: pd.DataFrame):
        self.model.fit(historical_data)

    def predict(self, year: int, week: int) -> pd.Series:
        return self.model.predict(year, week)
