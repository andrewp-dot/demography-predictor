# Standard library imports
import os
import pandas as pd
import logging
from typing import List, Dict, Literal, Optional
from torch import nn

import matplotlib.pyplot as plt


# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.utils.constants import (
    basic_features,
    highly_correlated_features,
    aging_targets,
    population_total_targets,
    gender_distribution_targets,
)

from src.pipeline import LocalModelPipeline
from src.train_scripts.train_local_models import (
    train_base_rnn,
    train_ensemble_model,
    train_arima_ensemble_model,
)

from src.base import TrainingStats

from model_experiments.base_experiment import BaseExperiment
from src.base import RNNHyperparameters

from src.compare_models.compare import ModelComparator

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


class FirstModelExperiment(BaseExperiment):

    SAVE_MODEL_DIR: str = os.path.abspath(
        os.path.join(".", "model_selection", "trained_models")
    )

    MODEL_NAMES: List[str] = [
        "ensemble-arima",
        "ensemble-arimax",
        "simple-rnn",
        "base-lstm",
        "base-gru",
        "xgboost",
        "rf",
        "lightgbm",
    ]

    # Need to save this to save their training stats for plot
    RNN_NAMES = ["simple-rnn", "base-gru", "base-lstm"]

    def __init__(
        self,
        description: str,
    ):
        super().__init__(name=self.__class__.__name__, description=description)
        self.FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

        # Get targets by experiment
        self.TARGETS: List[str] = self.FEATURES

        self.BASE_RNN_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=64,
            batch_size=16,
            output_size=len(self.TARGETS),
            epochs=30,
        )

        # EVALUATION_STATES: List[str] = ["Czechia", "Honduras", "United States"]
        # If empty select all states
        self.EVALUATION_STATES: List[str] = None  # If None, select all

        self.rnn_training_stats: Dict[str, TrainingStats] = {}
