# Standard library imports
import pandas as pd
import logging
from typing import Dict, Union, List, Optional
import torch

# Custom library imports
from src.utils.log import setup_logging
from src.utils.save_model import save_model, get_model

from src.base import RNNHyperparameters

from src.feature_model.model import BaseRNN
from src.feature_model.finetunable_model import FineTunableLSTM
from src.statistical_models.arima import CustomARIMA

from src.preprocessors.data_transformer import DataTransformer

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

# Get logger
logger = logging.getLogger("local_model")


# TODO: add hyperparameters or something...
# TODO: fix this -> make pure ensemble model compatibile for pipeline
class PureEnsembleModel:

    def __init__(
        self,
        target_models: Dict[str, Union[CustomARIMA, BaseRNN, FineTunableLSTM]],
        features: Optional[List[str]] = None,
    ):

        # Get features
        if not features:
            features = []

        self.FEATURES: List[str] = features

        # Targets are the same
        self.TARGETS: List[str] = list(target_models.keys())

        self.model: Dict[str, Union[CustomARIMA, BaseRNN, FineTunableLSTM]] = (
            target_models
        )

    # TODO: put the tensor on the same device as the model is.
    def predict(
        self, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        # Preprocess data
        years_to_predict = range(last_year + 1, target_year + 1)

        # timestep_predictions: pd.DataFrame | None = None
        timestep_predictions: torch.Tensor | None = None

        # Predict each feature for every year
        for target in self.TARGETS:

            current_model = self.model[target]
            # By type
            if isinstance(current_model, CustomARIMA):

                target_prediction_df = current_model.predict(
                    data=input_data, steps=len(years_to_predict)
                )

                # TODO: make the tensor from this
                target_prediction = torch.tensor(
                    target_prediction_df.values, dtype=torch.float32
                )

            if isinstance(current_model, BaseRNN) or isinstance(
                current_model, FineTunableLSTM
            ):
                # Predict featue for X years
                target_prediction = current_model.predict(
                    input_data=input_data[target].to_frame(),
                    last_year=last_year,
                    target_year=target_year,
                )

            # Rows - years
            # Columns -> features
            if timestep_predictions is None:
                # timestep_predictions = target_prediction_df
                timestep_predictions = target_prediction
            else:
                # Concat by columns
                timestep_predictions = torch.cat(
                    [timestep_predictions, target_prediction], dim=1
                )

        # At the end, after the loop
        prediction_df = pd.DataFrame(
            data=timestep_predictions.numpy(),  # or .cpu().numpy() if tensor is on GPU
            columns=self.TARGETS,
        )
        return prediction_df


#     [{'best_model': 'BaseRNN', 'feature': 'fertility rate, total'},
#  {'best_model': 'BaseRNN', 'feature': 'population, total'},
#  {'best_model': 'ARIMA', 'feature': 'net migration'},
#  {'best_model': 'BaseRNN', 'feature': 'arable land'},
#  {'best_model': 'BaseRNN', 'feature': 'birth rate, crude'},
#  {'best_model': 'ARIMA', 'feature': 'gdp growth'},
#  {'best_model': 'ARIMA', 'feature': 'death rate, crude'},
#  {'best_model': 'BaseRNN', 'feature': 'agricultural land'},
#  {'best_model': 'ARIMA', 'feature': 'rural population'},
#  {'best_model': 'ARIMA', 'feature': 'rural population growth'},
#  {'best_model': 'BaseRNN', 'feature': 'age dependency ratio'},
#  {'best_model': 'ARIMA', 'feature': 'urban population'},
#  {'best_model': 'BaseRNN', 'feature': 'population growth'},
#  {'best_model': 'BaseRNN', 'feature': 'adolescent fertility rate'},
#  {'best_model': 'BaseRNN', 'feature': 'life expectancy at birth, total'}]
