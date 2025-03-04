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


from config import setup_logging, Config
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from statsmodels.tsa.arima.model import ARIMA

# Custom imports
from local_model_benchmark.experiments.data_experiments import (
    whole_dataset_experiment,
    single_state_data_experiment,
    only_stationary_data_experiment,
)
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateModel


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: Data preprocessing functions -> split-> scaling -> sequences -> batches
# TODO: doc comments


# Data based experiments
def run_data_experiments() -> None:
    single_state_data_experiment(state="Czechia", split_rate=0.8)
    # whole_dataset_experiment()
    # only_stationary_data_experiment(state="Czechia", split_rate=0.8)


# Mode settings and comparision experiments
def run_model_experiments() -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    run_data_experiments()

    # run_model_experiments()


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
