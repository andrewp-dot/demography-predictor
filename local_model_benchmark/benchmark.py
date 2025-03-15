"""
In this file are experiments with local model.
"""

# Standard libraries
import os
import pandas as pd
import pprint
import logging
import torch
from typing import List, Tuple, Union


from config import Config
from src.utils.log import setup_logging


# Custom imports
from local_model_benchmark.experiments.data_experiments import OneStateDataExperiment
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: doc comments

# Here create and define experiments

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
