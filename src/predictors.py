# Standard library imports
import pandas as pd
from typing import Union, List
from xgboost import XGBRegressor

# Custom imports
from config import setup_logging
from src.local_model.model import LSTMHyperparameters, LocalModel
from src.global_model.model import GlobalModel
from src.local_model.statistical_models import LocalARIMA


# TODO: implement different models
# TODO: RNN + XBOOST? -> model v1
# TODO: ARIMA + XGBOOST -> model v2
class DemographyPredictor:

    def __init__(
        self,
        name: str,
        local_model: Union[LocalModel, LocalARIMA],
        global_model: Union[GlobalModel],
    ):
        # Define name of the model -> name is used to versionning etc
        self.name: str = name

        # Define architecture: local model -> global model
        self.local_model: Union[LocalModel, LocalARIMA] = local_model
        self.global_model: Union[GlobalModel] = global_model

    def __repr__(self) -> str:
        return f"{self.name} ({type(self.local_model).__name__} -> {type(self.global_model.model).__name__})"

    def predict(self, input_data: pd.DataFrame):
        raise NotImplementedError("Cannoct predict")


def predictor_v1() -> DemographyPredictor:

    # Define features
    FEATURES: List[str] = ["year"]
    TARGET: str = "population, total"

    # Define local model
    hyperparameters = LSTMHyperparameters(
        input_size=len(FEATURES + [TARGET]),
        hidden_size=2048,
        sequence_length=15,
        learning_rate=0.0001,
        epochs=30,
        batch_size=1,
        num_layers=4,
    )

    local_model = LocalModel(hyperparameters=hyperparameters)

    # Define global model
    global_model: GlobalModel = GlobalModel(model=XGBRegressor())

    # Create predictor
    predictor = DemographyPredictor(
        "predictor_v1", local_model=local_model, global_model=global_model
    )

    return predictor


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Create predictor
    pred = predictor_v1()

    # Print predictor
    print(pred)
