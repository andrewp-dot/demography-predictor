# Standard libraries
import os
import pandas as pd
import pprint
import logging
import torch
from typing import List, Dict, Literal

import copy


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

# Get settings
settings = Config()

# TODO: plot this data and so on...


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
class OptimalParamsExperiment(BaseExperiment):

    # TODO: evaluation -> use more then r2 score?

    def adjust_hidden_size(
        self, base_parameters: LSTMHyperparameters, hidden_size: int
    ) -> LSTMHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.hidden_size = hidden_size
        return new_params

    def adjust_sequence_len(
        self, base_parameters: LSTMHyperparameters, sequence_length: int
    ) -> LSTMHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.sequence_length = sequence_length
        return new_params

    def adjust_num_layers(
        self, base_parameters: LSTMHyperparameters, num_layers: int
    ) -> LSTMHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.num_layers = num_layers
        return new_params

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
            rnn.train_model(
                batch_inputs=train_batches,
                batch_targets=target_batches,
                display_nth_epoch=2,
            )

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
            f"[Optimal parameter]: Optimal {parameter_name}: {optimal_parameter}, Evaluation:\n{formatted_last_model_evaluation}"
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
        # Create readme
        self.create_readme()

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

        # Compare basic model and model with found parameters:

        # Train and evaluate base model
        base_train_batches, base_target_batches, base_scaler = (
            preprocess_single_state_data(
                train_data_df=train_data_df,
                state_loader=state_loader,
                hyperparameters=BASE_HYPERPARAMS,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        base_model = LocalModel(hyperparameters=BASE_HYPERPARAMS)

        base_model.train_model(
            batch_inputs=base_train_batches,
            batch_targets=base_target_batches,
            display_nth_epoch=2,
        )

        base_model_evaluation = EvaluateLSTM(base_model)
        base_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
            features=FEATURES,
            scaler=base_scaler,
        )

        # Plot and save base model plot
        base_fig = base_model_evaluation.plot_predictions()

        self.readme_add_section(
            title="# Base model evaluation",
            text=f"Hyperparameters:\n```{str(BASE_HYPERPARAMS)}```",
        )

        self.save_plot(fig_name="base_model_eval.png", figure=base_fig)
        self.readme_add_plot(
            plot_name="Base model predicted vs reference values",
            plot_description="Displays the performance for every feature predicted of the `Base Model`.",
            fig_name="base_model_eval.png",
        )

        # Train and evaluate parametricaly adjusted model
        # Rewrite the parameters to optimal
        OPTIMAL_PAREMETRS: LSTMHyperparameters = BASE_HYPERPARAMS
        OPTIMAL_PAREMETRS.hidden_size = found_optimal_paremeters["hidden_size"]
        OPTIMAL_PAREMETRS.sequence_length = found_optimal_paremeters["sequence_length"]
        OPTIMAL_PAREMETRS.num_layers = found_optimal_paremeters["num_layers"]

        # Preprocess data
        optimal_train_batches, optimal_target_batches, optimal_scaler = (
            preprocess_single_state_data(
                train_data_df=train_data_df,
                state_loader=state_loader,
                hyperparameters=OPTIMAL_PAREMETRS,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        optimal_model = LocalModel(hyperparameters=OPTIMAL_PAREMETRS)

        optimal_model.train_model(
            batch_inputs=optimal_train_batches,
            batch_targets=optimal_target_batches,
            display_nth_epoch=2,
        )

        optimal_model_evaluation = EvaluateLSTM(optimal_model)
        optimal_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
            features=FEATURES,
            scaler=optimal_scaler,
        )

        # Plot and save base model plot
        optimal_fig = optimal_model_evaluation.plot_predictions()

        self.save_plot(fig_name="optimal_model_eval.png", figure=optimal_fig)

        self.readme_add_section(
            title="# Optimal model evaluation",
            text=f"Hyperparameters:\n```{str(OPTIMAL_PAREMETRS)}```",
        )

        self.readme_add_plot(
            plot_name="Optimal model predicted vs reference values",
            plot_description="Displays the performance for every feature predicted of the `Optimal Model`.",
            fig_name="optimal_model_eval.png",
        )

        # Train and evaluate the adjusted parameters model
        for param_name, value in found_optimal_paremeters.items():

            print(f"Optimal {param_name} is: {value}")

        # Save the results
        formatted_base_model_evaluation: str = pprint.pformat(
            base_model_evaluation.to_readable_dict()
        )
        formatted_optimal_model_evaluation: str = pprint.pformat(
            optimal_model_evaluation.to_readable_dict()
        )
        compare_models_by_metric: str = f"""
Base model:
{formatted_base_model_evaluation}

Optimal model:
{formatted_optimal_model_evaluation}
"""

        self.readme_add_section(
            title="# Compare metric results", text=compare_models_by_metric
        )


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
    exp = OptimalParamsExperiment(
        name="OptimalParamsExperiment",
        description="The goal is to find the optimal parameters for the given LocalModel model.",
    )

    # Run
    exp.run(state="Czechia", split_rate=0.8)
