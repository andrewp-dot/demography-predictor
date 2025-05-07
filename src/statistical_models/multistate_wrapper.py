# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import pandas as pd
from typing import List, Dict
import joblib

# Custom imports
from src.feature_model.ensemble_model import PureEnsembleModel


# TODO: add LazyLoading in here
class StatisticalMultistateWrapper:
    """
    Wrapper for training multiple models for each sequence (because of statistical model approach).
    """

    def __init__(
        self,
        # model: Dict[str, str],
        model: Dict[str, PureEnsembleModel],
        features: List[str],
        targets: List[str],
    ):
        # Define model
        self.model: Dict[str, str] = model
        self._loaded_models: Dict[str, PureEnsembleModel] = {}

        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

    # def lazy_load_model(self, state: str) -> PureEnsembleModel:
    #     """
    #     Lazy load the model for a given state.
    #     """
    #     # Check if the model is already loaded
    #     if state in self._loaded_models:
    #         return self._loaded_models[state]

    #     # Load the model
    #     model = joblib.load(self.model[state])

    #     # Store the model in the dictionary
    #     self._loaded_models[state] = model

    #     return model

    # def save_loaded_models(self):
    #     """
    #     Save the loaded models to disk.
    #     """
    #     for state, model in self._loaded_models.items():
    #         joblib.dump(model, self.model[state])

    def predict(
        self, state: str, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        # Get model by state
        # model = self.lazy_load_model(state=state)
        model = self.model[state]

        prediction_df = model.predict(
            input_data=input_data, last_year=last_year, target_year=target_year
        )

        return prediction_df
