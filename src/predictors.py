# Standard library imports
import pandas as pd
from typing import Union, List
from xgboost import XGBRegressor

# Custom imports
from src.local_model.model import LocalModel
from src.local_model.statistical_models import ARIMA


# TODO: implement different models
# TODO: RNN + XBOOST? -> model v1
# TODO: ARIMA + XGBOOST -> model v2
class DemographyPredictor:

    def __init__(
        self,
        name: str,
        local_model: Union[LocalModel, ARIMA],
        global_model: Union[XGBRegressor],
    ):
        # Define name of the model -> name is used to versionning etc
        self.name: str = name

        # Define architecture: local model -> global model
        self.local_model: Union[LocalModel, ARIMA] = local_model
        self.global_model: Union[XGBRegressor] = global_model

    def __repr__(self) -> str:
        return f"{self.name} ({type(self.local_model).__name__} -> {type(self.global_model).__name__})"

    def predict(self, input_data: pd.DataFrame):
        raise NotImplementedError("Cannoct predict")


class DemographyPredictorV1:

    def __init__(self):
        pass


class DemographyPRedictorV2:

    def __init__(self):
        pass
