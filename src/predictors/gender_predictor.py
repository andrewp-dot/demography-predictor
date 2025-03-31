# Standard library imports
import logging
from typing import Union, List
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.utils.save_model import save_model

from src.predictors.predictor_base import create_local_model
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.state_preprocessing import StateDataLoader

# Get logger and settings
logger = logging.getLogger("demograpy_predictor")
settings = Config()

# Get all posible targete features to exclude them from the input
ALL_POSSIBLE_TARGETS = settings.ALL_POSSIBLE_TARGET_FEATURES


if __name__ == "__main__":
    pass
