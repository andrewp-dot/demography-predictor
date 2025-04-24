# TODO write global model selection experiment


# Standard library imports
import logging
from typing import List, Dict
from torch import nn

# Import tested tree models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.utils.constants import (
    basic_features,
    hihgly_correlated_features,
    aging_targets,
)

from src.pipeline import GlobalModelPipeline
from src.train_scripts.train_global_models import (
    train_global_model_tree,
    train_global_rnn,
)

from src.global_model.model import XGBoostTuneParams

from model_experiments.base_experiment import BaseExperiment
from src.base import RNNHyperparameters

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger("benchmark")


class SecondModelSelection(BaseExperiment):
    """
    Question: How to evaluate this? GROUND TRUTH testing? For now YES.
    """

    FEATURES: List[str] = basic_features(exclude=hihgly_correlated_features)

    TARGETS: List[str] = aging_targets()

    BASE_RNN_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES + TARGETS),
        hidden_size=256,
        batch_size=16,
        output_size=len(TARGETS),
    )

    XGBOOST_TUNE_PARAMETERS: XGBoostTuneParams = XGBoostTuneParams(
        n_estimators=[200, 400],
        learning_rate=[0.01, 0.05, 0.1],
        max_depth=[3, 5, 7],
        subsample=[0.8, 1.0],
        colsample_bytree=[0.8, 1.0],
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def run(self, split_rate: float = 0.8) -> None:

        # Create readme
        self.create_readme()
        DISPLAY_NTH_EPOCH = 1

        # Load data
        loader = StatesDataLoader()
        states_data_dict = loader.load_all_states()

        TO_COMPARE_PIPELINES: Dict[str, GlobalModelPipeline] = {}

        # Train classic rnn
        logger.info("Training simple rnn...")
        TO_COMPARE_PIPELINES["simple-rnn"] = train_global_rnn(
            name="simple-rnn",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=DISPLAY_NTH_EPOCH,
            rnn_type=nn.RNN,
        )

        # Train lstm
        logger.info("Training base lstm...")
        TO_COMPARE_PIPELINES["base-lstm"] = train_global_rnn(
            name="base-lstm",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=DISPLAY_NTH_EPOCH,
            rnn_type=nn.LSTM,
        )

        # Train gru
        logger.info("Training base gru...")
        TO_COMPARE_PIPELINES["base-gru"] = train_global_rnn(
            name="base-gru",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=DISPLAY_NTH_EPOCH,
            rnn_type=nn.GRU,
        )

        # Train xgboost
        logger.info("Training xgboost...")
        TO_COMPARE_PIPELINES["xgboost"] = train_global_model_tree(
            name="xgboost",
            tree_model=XGBRegressor(objective="reg:squarederror", random_state=42),
            states_data=states_data_dict,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            tune_parameters=self.XGBOOST_TUNE_PARAMETERS,
        )

        # Train randomforest
        logger.info("Training random forest...")
        TO_COMPARE_PIPELINES["rf"] = train_global_model_tree(
            name="rf",
            tree_model=RandomForestRegressor(n_estimators=100, random_state=42),
            states_data=states_data_dict,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            tune_parameters=self.XGBOOST_TUNE_PARAMETERS,
        )

        # Train lightgbm - maybe?

        # Train arima


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    exp = SecondModelSelection("test")
    exp.run(split_rate=0.8)
