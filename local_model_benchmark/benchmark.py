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
from local_model_benchmark.data_experiments import (
    whole_dataset_experiment,
    single_state_data_experiment,
    only_stationary_data_experiment,
)
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateLSTM


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: Data preprocessing functions -> split-> scaling -> sequences -> batches
# TODO: doc comments


## Maybe here define base experiment function


def run_data_experiments() -> None:
    # Data experiments
    single_state_data_experiment(state="Czechia", split_rate=0.8)
    # whole_dataset_experiment()
    # only_stationary_data_experiment(state="Czechia", split_rate=0.8)


def run_model_experiments() -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    run_data_experiments()

    # run_model_experiments()


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
# 2. Compare prediction using whole state data and the last few records of data
# 3. Predict parameters for different years (e.g. to 2030, 2040, ... )
# 4. Compare model with statistical methods (ARIMA, GM)


def compare_with_statistical_models(state: str, split_rate: float) -> None:

    # TODO: implement ARIMA
    # TODO: implement grey model
    # TODO: train rnn
    # TODO: compare them
    raise NotImplementedError("")


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
