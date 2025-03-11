# Standard library imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.predictors import DemographyPredictor


class ModelPipeline:

    def __init__(self, model: DemographyPredictor, fitted_scaler: MinMaxScaler):
        self.model: DemographyPredictor = model
        self.fitted_scaler: MinMaxScaler = fitted_scaler

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        self.model.predict(input_data=data)
        raise NotImplementedError("Model pipeline is not implemented yet.")


raise NotImplementedError("Model pipeline is not implemented yet.")
