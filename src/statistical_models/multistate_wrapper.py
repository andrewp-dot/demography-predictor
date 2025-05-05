# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import pandas as pd
from typing import List, Dict

# Custom imports

from src.feature_model.ensemble_model import PureEnsembleModel


class StatisticalMultistateWrapper:
    """
    Wrapper for training multiple models for each sequence (because of statistical model approach).
    """

    def __init__(
        self,
        model: Dict[str, PureEnsembleModel],
        features: List[str],
        targets: List[str],
    ):
        # Define model
        self.model: Dict[str, PureEnsembleModel] = model
        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

    def predict(
        self, state: str, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        # Get model by state
        model = self.model[state]

        prediction_df = model.predict(
            input_data=input_data, last_year=last_year, target_year=target_year
        )

        return prediction_df
