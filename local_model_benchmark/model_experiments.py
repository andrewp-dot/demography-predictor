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


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
