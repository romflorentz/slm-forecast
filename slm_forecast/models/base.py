from enum import Enum, auto

import pandas as pd


class ModelType(Enum):
    RATIO_BASED = auto()
    LINEAR = auto()
    RIDGE = auto()


class BaseModel:
    name: ModelType

    def fit(self, historical_data: pd.DataFrame):
        pass

    def predict(self, year: int, week: int) -> pd.Series:
        pass


def create_model(model_type: ModelType):
    if model_type == ModelType.RATIO_BASED:
        from slm_forecast.models.ratio_based import RatioBasedModel

        return RatioBasedModel()
    elif model_type == ModelType.LINEAR:
        from slm_forecast.models.linear import LinearRegressionModel

        return LinearRegressionModel()
    elif model_type == ModelType.RIDGE:
        from slm_forecast.models.ridge import RidgeRegressionModel

        return RidgeRegressionModel()
    else:
        raise ValueError(f"Model type {model_type} not supported.")
