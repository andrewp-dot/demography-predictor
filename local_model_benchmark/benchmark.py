"""
In this file are experiments with local model.
"""

# Standard libraries
import logging
from typing import Dict, List


# Custom imports
from config import Config
from src.utils.log import setup_logging

from local_model_benchmark.experiments.data_experiments import OneStateDataExperiment
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel

from local_model_benchmark.experiments.base_experiment import Experiment

settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: doc comments


# List of available experimets
experiment_list: List[Experiment] = []

# Setup experiments -> convert experiment list to dict
experiments: Dict[str, Experiment] = {exp.exp.name: exp for exp in experiment_list}


# if __name__ == "__main__":
# experiments()


# Use different scaler(s)
# 1. Robust scalers, MinMax, Logaritmic transformation

# Odstranit outliers?
# Zaokrúhlenie dát, čo s nimi?

# Spájanie modelov:
# Stacking? - priemerovanie vysledkov viacerych modelov subezne
# Boosting? - Ada boost (les neuroniek? :D) , XGBoost

# TODO: try different loss functions

#  criterion = nn.HuberLoss(delta=1.0)

# Interesting to try:
# Feature engineering -> correlations, stationary features, non-stationary features
# Maybe delete the COVID years in order to known it's influence on model accuracy?
