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
from local_model_benchmark.utils import preprocess_single_state_data
from local_model_benchmark.experiments.base_experiment import BaseExperiment

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateLSTM


# Setup logger
logger = logging.getLogger("benchmark")


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
class OptimalParamsExperiment(BaseExperiment):

    # Find optimal neuron number in layer number (hidden_size)
    def find_optimal_hidden_size(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        state_loader: StateDataLoader,
        base_params: LSTMHyperparameters,
        features: List[str],
        possible_sizes: List[int],
    ) -> int:

        # Set features as constant
        FEATURES = features

        # Current hyperparameters
        current_hyperparams = base_params

        # Init the optimal parameter
        optimal_parameter: int = current_hyperparams.hidden_size

        # Init last model score
        last_model_r2_score: float | None = None
        for size in possible_sizes:

            # Set the size of the desired
            current_hyperparams.hidden_size = size

            # Preprocess data for the desired neural network
            train_batches, target_batches, state_scaler = preprocess_single_state_data(
                train_data_df=train_df,
                state_loader=state_loader,
                hyperparameters=current_hyperparams,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )

            # Train the model
            rnn = LocalModel(current_hyperparams)
            rnn.train_model(batch_inputs=train_batches, batch_targets=target_batches)

            # Evaluate model
            rnn_evaluation = EvaluateLSTM(rnn)

            rnn_evaluation.eval(
                test_X=train_df, test_y=test_df, features=FEATURES, scaler=state_scaler
            )

            # Get overall metrics
            current_model_r2_score = rnn_evaluation.overall_metrics.loc["r2"].item()

            # If there is just one model save the score
            if last_model_r2_score is None:
                last_model_r2_score = current_model_r2_score
                continue

            # Compare the model score with last score
            if last_model_r2_score < current_model_r2_score:
                optimal_parameter = size

        return optimal_parameter

    # Find optimal sequence length
    def find_optimal_sequence_len(
        self, train_df: pd.DataFrame, base_params: LSTMHyperparameters, range: range
    ) -> int:
        raise NotImplementedError()

    # Find optimal learning rate
    def find_optimal_learning_rate(
        self,
        train_df: pd.DataFrame,
        base_params: LSTMHyperparameters,
    ) -> float:
        raise NotImplementedError()

    # Find optimal number of layers
    def find_optimal_number_of_layers(
        self, train_df: pd.DataFrame, base_params: LSTMHyperparameters, range: range
    ) -> int:
        raise NotImplementedError()

    def run(self, state: str, split_rate: float) -> None:
        # Load data
        STATE = state
        state_loader = StateDataLoader(STATE)

        state_df = state_loader.load_data()

        # Drop country name
        state_df.drop(columns=["country name"], inplace=True)

        # Get features
        FEATURES = [col.lower() for col in state_df.columns]

        BASE_HYPER_PARAMS = LSTMHyperparameters(
            input_size=len(FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )

        # Split data
        train_data_df, test_data_df = state_loader.split_data(
            state_df, split_rate=split_rate
        )

        # Find optimal params
        optim_hidden_size = self.find_optimal_hidden_size(
            possible_sizes=[8, 16, 32, 64, 128, 256, 512]
        )
        optim_seq_len = self.find_optimal_sequence_len(range=range(15))
        optim_learning_rate = self.find_optimal_learning_rate(
            base_learning_rate=BASE_HYPER_PARAMS.learning_rate
        )
        optim_layer_num = self.find_optimal_number_of_layers(range=range(3, 7))

        # Save the results

        # Try to train model with optimal parameters


# 2. Compare model with statistical methods (ARIMA, GM)
# 2.1. VAR, SARIMA, ARIMA * 19?
class StatisticalModelsExperiment(BaseExperiment):
    raise NotImplementedError(
        "Need to implement and compare the experiment with the model!"
    )


# 3. Compare prediction using whole state data and the last few records of data
# 4. Predict parameters for different years (e.g. to 2030, 2040, ... )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
