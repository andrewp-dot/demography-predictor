# Standard libraries
import os
import pandas as pd
import pprint
import logging
import torch
from typing import List, Tuple, Union


from config import setup_logging, Config
from local_model_benchmark.utils import (
    preprocess_single_state_data,
    write_experiment_results,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from statsmodels.tsa.arima.model import ARIMA

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateLSTM

# TODO: define experiments

# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
# 2. Compare prediction using whole state data and the last few records of data
# 3. Predict parameters for different years (e.g. to 2030, 2040, ... )
# 4. Compare model with statistical methods (ARIMA, GM)
# 4.1. VAR, SARIMA, ARIMA * 19?


def compare_with_statistical_models(state: str, split_rate: float) -> None:

    # TODO: implement ARIMA
    # TODO: implement grey model
    # TODO: train rnn
    # TODO: compare them
    raise NotImplementedError("")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
