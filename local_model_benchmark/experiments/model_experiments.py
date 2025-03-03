# Standard libraries
import os
import pandas as pd
import pprint
import logging
import torch
from typing import List, Dict, Literal


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

# TODO: plot this data and so on...


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
class OptimalParamsExperiment(BaseExperiment):

    # TODO: evaluation -> use more then r2 score?

    def adjust_hidden_size(
        self, base_parameters: LSTMHyperparameters, hidden_size: int
    ) -> LSTMHyperparameters:
        base_parameters.hidden_size = hidden_size
        return base_parameters

    def adjust_sequence_len(
        self, base_parameters: LSTMHyperparameters, sequence_length: int
    ) -> LSTMHyperparameters:
        base_parameters.sequence_length = sequence_length
        return base_parameters

    def adjust_num_layers(
        self, base_parameters: LSTMHyperparameters, num_layers: int
    ) -> LSTMHyperparameters:
        base_parameters.num_layers = num_layers
        return base_parameters

    # Find optimal neuron number in layer number (hidden_size)
    def find_optimal_parameter(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        state_loader: StateDataLoader,
        base_params: LSTMHyperparameters,
        features: List[str],
        parameter_name: Literal["hidden_size", "sequence_length", "num_layers"],
        parameter_options: List[int],
    ) -> int:

        # Set features as constant
        FEATURES = features

        # Current hyperparameters
        current_hyperparams = base_params

        # Init the optimal parameter and parameter setting function
        # optimal_parameter: int | float | None = None

        if parameter_name == "hidden_size":
            optimal_parameter = current_hyperparams.hidden_size
            set_parameter = self.adjust_hidden_size
        elif parameter_name == "sequence_length":
            optimal_parameter = current_hyperparams.sequence_length
            set_parameter = self.adjust_sequence_len
        elif parameter_name == "num_layers":
            optimal_parameter = current_hyperparams.num_layers
            set_parameter = self.adjust_num_layers
        else:
            raise ValueError(
                f"Cannot find the optimal parameter for parameter '{parameter_name}'. This parameter is not supported!"
            )

        # Init last model score
        last_model_evaluation: EvaluateLSTM | None = None

        all_evaluations: List[EvaluateLSTM] = []
        for parameter_option in parameter_options:

            # Update the parameter to the next option
            base_params = set_parameter(base_params, parameter_option)

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

            # Get r2 scorre from overall metrics
            all_evaluations.append(rnn_evaluation)

            # If there is just one model save the score
            if last_model_evaluation is None:
                last_model_evaluation = rnn_evaluation
                continue

            # Compare the model score with last score
            if last_model_evaluation.is_new_better(new_model_evaluation=rnn_evaluation):
                last_model_evaluation = rnn_evaluation
                optimal_parameter = parameter_option

        # Get the best model evaluation
        formatted_last_model_evaluation = pprint.pformat(
            last_model_evaluation.to_readable_dict()
        )
        logger.info(
            f"[Optimal hidden size]: Optimal hidden size: {optimal_parameter}, Evaluation:\n{formatted_last_model_evaluation}"
        )

        all_evaluations_dict: str = {
            size: evaluation.to_readable_dict()
            for size, evaluation in zip(parameter_options, all_evaluations)
        }

        # Get all models evaluation
        formatted_all_models_evaluation: str = pprint.pformat(all_evaluations_dict)
        logger.info(f"[All evaluation of sizes]:\n{formatted_all_models_evaluation}")
        return optimal_parameter

    # Find optimal learning rate
    def find_optimal_learning_rate(
        self,
        train_df: pd.DataFrame,
        base_params: LSTMHyperparameters,
    ) -> float:
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

        BASE_HYPERPARAMS = LSTMHyperparameters(
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
        optimal_parameters_settings: Dict[str, List[int | float]] = {
            "hidden_size": [8, 16, 32, 64, 128, 256, 512],
            "sequence_length": list(range(3, 12)),
            "num_layers": list(range(3, 7)),
        }

        # Save the found optimal parameters
        found_optimal_paremeters: Dict[str, int | float] = {}

        for param_name, param_options in optimal_parameters_settings.items():
            optimal_param = self.find_optimal_parameter(
                train_df=train_data_df,
                test_df=test_data_df,
                state_loader=state_loader,
                base_params=BASE_HYPERPARAMS,
                features=FEATURES,
                parameter_name=param_name,
                parameter_options=param_options,
            )

            found_optimal_paremeters[param_name] = optimal_param

        # Print found parameters
        print()
        print("-" * 100)

        for param_name, value in found_optimal_paremeters.items():

            print(f"Optimal {param_name} is: {value}")

        # Save the results

        # Try to train model with optimal parameters


# 2. Compare model with statistical methods (ARIMA, GM)
# 2.1. VAR, SARIMA, ARIMA * 19?
# class StatisticalModelsExperiment(BaseExperiment):
#     raise NotImplementedError(
#         "Need to implement and compare the experiment with the model!"
#     )


# 3. Compare prediction using whole state data and the last few records of data
# 4. Predict parameters for different years (e.g. to 2030, 2040, ... )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    OptimalParamsExperiment().run(state="Czechia", split_rate=0.8)
