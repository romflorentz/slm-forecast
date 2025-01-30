import os
from warnings import simplefilter

import fire
import numpy as np
import pandas as pd

from slm_forecast.evaluate import evaluate
from slm_forecast.models.base import ModelType, create_model
from slm_forecast.plot import plot
from slm_forecast.train import fit_predict

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Folders and files
DATA_RAW_FOLDER = "data/raw"
DATA_PROCESSED_FOLDER = "data/processed"
DATA_MODELS_FOLDER = "data/models"
DATA_PREDICTIONS_FOLDER = "data/predictions"
DATA_EVALUATIONS_FOLDER = "data/evaluations"
DATA_PLOTS_FOLDER = "data/plots"
SALES_RAW_DATA_PATH = os.path.join(DATA_RAW_FOLDER, "sales.csv")
SALES_PROCESSED_DATA_PATH = os.path.join(DATA_PROCESSED_FOLDER, "sales.csv")
SALES_PREDICTION_PATH = os.path.join(DATA_PREDICTIONS_FOLDER, "predictions.csv")

EXCLUDED_CATEGORIES = ["FTW GRAVEL RUNNING"]


class SalesForecastApp:
    def __init__(self, model_type: str | ModelType = ModelType.RATIO_BASED):
        self.model_type = (
            model_type
            if isinstance(model_type, ModelType)
            else ModelType[model_type.upper()]
        )
        self.model = create_model(self.model_type)

    def process(
        self, data_path=SALES_RAW_DATA_PATH, output_path=SALES_PROCESSED_DATA_PATH
    ):
        df = pd.read_csv(data_path)
        df["WEEKNUM"] = pd.to_datetime(df["WEEK"]).dt.isocalendar().week
        df["YEAR"] = pd.to_datetime(df["WEEK"]).dt.isocalendar().year
        df.drop(
            columns=["WEEK"] + EXCLUDED_CATEGORIES,
            inplace=True,
        )
        df.set_index(["YEAR", "WEEKNUM"], inplace=True)
        df = np.log1p(df)
        df.reset_index(inplace=True)
        df.to_csv(output_path, index=False)

    def fit_predict(
        self,
        data_path=SALES_PROCESSED_DATA_PATH,
        start_year=2024,
        start_week=1,
        n_weeks=52,
    ):
        os.makedirs(DATA_PREDICTIONS_FOLDER, exist_ok=True)
        df = pd.read_csv(data_path)
        predictions_df = fit_predict(self.model, df, start_year, start_week, n_weeks)
        predictions_df.to_csv(SALES_PREDICTION_PATH, index=False)

    def plot(
        self, data_path=SALES_PROCESSED_DATA_PATH, predict_path=SALES_PREDICTION_PATH
    ):
        os.makedirs(DATA_PLOTS_FOLDER, exist_ok=True)
        self.fit_predict(data_path)
        df_hist = pd.read_csv(data_path)
        df_pred = pd.read_csv(predict_path)
        output_path = f"{DATA_PLOTS_FOLDER}/predictions_{self.model_type.name}.png"
        plot(df_hist, df_pred, self.model_type, output_path)

    def evaluate(
        self, data_path=SALES_PROCESSED_DATA_PATH, predict_path=SALES_PREDICTION_PATH
    ):
        os.makedirs(DATA_EVALUATIONS_FOLDER, exist_ok=True)
        self.fit_predict(data_path)
        df_hist = pd.read_csv(data_path)
        df_pred = pd.read_csv(predict_path)
        metrics_df = evaluate(df_hist, df_pred)
        os.makedirs(DATA_EVALUATIONS_FOLDER, exist_ok=True)
        model_name = self.model_type.name.lower()
        output_path = os.path.join(DATA_EVALUATIONS_FOLDER, f"{model_name}.csv")
        metrics_df.to_csv(output_path)
        print(f"Metrics saved to {output_path}")

    def evaluate_all(self, data_path=SALES_PROCESSED_DATA_PATH):
        for model_type in ModelType:
            self.model_type = model_type
            self.model = create_model(model_type)
            self.evaluate(data_path)


if __name__ == "__main__":
    fire.Fire(SalesForecastApp)
