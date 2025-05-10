# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import pandas as pd
import logging
from typing import Dict, Union, List, Optional
import torch

# Custom imports
from src.feature_model.model import BaseRNN
from src.feature_model.finetunable_model import FineTunableLSTM
from src.statistical_models.arima import CustomARIMA


# Get logger
logger = logging.getLogger("local_model")


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

        self.sequence_length: int = self.__get_sequence_length()

    def __get_sequence_length(self) -> int:
        """
        Get the sequence length of the model.

        Returns:
            out: int: The sequence length of the model.
        """

        for model in self.model.values():
            if isinstance(model, BaseRNN) or isinstance(model, FineTunableLSTM):
                return model.hyperparameters.sequence_length
            else:
                return 10

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
