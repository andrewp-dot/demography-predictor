# Standard libraries
import os
import pandas as pd
import pprint
import logging
import torch
from typing import List, Dict, Literal, Tuple

import copy
import optuna


from config import setup_logging, Config
from local_model_benchmark.utils import (
    preprocess_single_state_data,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from statsmodels.tsa.arima.model import ARIMA

# Custom imports
from local_model_benchmark.utils import preprocess_single_state_data
from local_model_benchmark.experiments.base_experiment import BaseExperiment

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateModel


# Setup logger
logger = logging.getLogger("benchmark")

# Get settings
settings = Config()

# TODO: plot this data and so on...


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
class OptimalParamsExperiment(BaseExperiment):

    # TODO: evaluation -> use more then r2 score?
    def __init__(
        self,
        name: str,
        description: str,
        hidden_size_range: Tuple[int, int],
        sequence_length_range: Tuple[int, int],
        num_layers_range: Tuple[int, int],
        learning_rate_range: Tuple[int, int],
    ):
        super().__init__(name, description)

        # Add ranges for parameters
        self.hidden_size_range: Tuple[int, int] = self.__min_max_tuple(
            hidden_size_range
        )
        self.sequence_length_range: Tuple[int, int] = self.__min_max_tuple(
            sequence_length_range
        )
        self.num_layers_range: Tuple[int, int] = self.__min_max_tuple(num_layers_range)
        self.learning_rate_range: Tuple[int, int] = self.__min_max_tuple(
            learning_rate_range
        )

    def __min_max_tuple(self, tup: Tuple[int, int]) -> Tuple[int, int]:
        """
        Adjust tuple to format (min, max). If the both numbers are equal,

        Args:
            tup (Tuple): _description_

        Returns:
            Tuple: _description_
        """
        # if the max value is first
        if tup[0] > tup[1]:
            return tup[1], tup[0]

        return tup

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

    def adjust_learning_rate(
        self, base_parameters: LSTMHyperparameters, learning_rate: float
    ):
        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.learning_rate = learning_rate
        return new_params

    def find_optimal_hyperparams(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        base_params: LSTMHyperparameters,
        state_loader: StateDataLoader,
        features: List[str],
    ):

        def objective(trial: optuna.Trial):

            # Get ranges from settings
            hidden_size = trial.suggest_int(
                "hidden_size",
                self.hidden_size_range[0],
                self.hidden_size_range[1],
            )
            sequence_length = trial.suggest_int(
                "sequence_length",
                self.sequence_length_range[0],
                self.sequence_length_range[1],
            )
            num_layers = trial.suggest_int(
                "num_layers",
                self.num_layers_range[0],
                self.num_layers_range[1],
            )
            learning_rate = trial.suggest_float(
                "learning_rate",
                self.learning_rate_range[0],
                self.learning_rate_range[1],
            )

            # Set hyperparameters
            NEW_HYPERPARAMS = LSTMHyperparameters(
                input_size=base_params.input_size,
                # Set new
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                learning_rate=learning_rate,
                num_layers=num_layers,
                # Get old
                epochs=base_params.epochs,
                batch_size=base_params.batch_size,
                bidirectional=base_params.bidirectional,
            )

            # Preprocess data
            train_batches, target_batches, state_scaler = preprocess_single_state_data(
                train_data_df=train_df,
                state_loader=state_loader,
                hyperparameters=NEW_HYPERPARAMS,
                features=features,
                scaler=MinMaxScaler(),
            )

            # Train model
            rnn = LocalModel(base_params)
            rnn.train_model(batch_inputs=train_batches, batch_targets=target_batches)

            # Evaluate model
            rnn_evaluation = EvaluateModel(rnn)
            rnn_evaluation.eval(
                test_X=train_df, test_y=test_df, features=features, scaler=state_scaler
            )

            # Return a score (maximize R²)
            return rnn_evaluation.overall_metrics.loc[
                rnn_evaluation.overall_metrics["metric"] == "r2", "value"
            ].values[0]

        # Run Bayesian Optimization
        study = optuna.create_study(direction="maximize")  # Maximize R² score
        study.optimize(objective, n_trials=20)

        # Best parameters
        print("Best parameters:", study.best_params)

        return study.best_params

    def run(self, state: str, split_rate: float, features: List[str]):

        self.create_readme()

        # Load data
        STATE = state
        state_loader = StateDataLoader(STATE)

        state_df = state_loader.load_data()

        # Drop country name
        state_df.drop(columns=["country name"], inplace=True)

        # Get features
        FEATURES = [col.lower() for col in features]

        BASE_HYPERPARAMS = LSTMHyperparameters(
            input_size=len(FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=20,
            batch_size=1,
            num_layers=3,
        )

        # Split data
        train_data_df, test_data_df = state_loader.split_data(
            state_df, split_rate=split_rate
        )

        best_params_dict = self.find_optimal_hyperparams(
            train_df=train_data_df,
            test_df=test_data_df,
            base_params=BASE_HYPERPARAMS,
            state_loader=state_loader,
            features=FEATURES,
        )

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

        base_model_evaluation = EvaluateModel(base_model)
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
        OPTIMAL_PAREMETRS = self.adjust_hidden_size(
            BASE_HYPERPARAMS, best_params_dict["hidden_size"]
        )
        OPTIMAL_PAREMETRS = self.adjust_sequence_len(
            OPTIMAL_PAREMETRS, best_params_dict["sequence_length"]
        )
        OPTIMAL_PAREMETRS = self.adjust_num_layers(
            OPTIMAL_PAREMETRS, best_params_dict["num_layers"]
        )
        OPTIMAL_PAREMETRS = self.adjust_learning_rate(
            OPTIMAL_PAREMETRS, best_params_dict["learning_rate"]
        )

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

        optimal_model_evaluation = EvaluateModel(optimal_model)
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
class CompareStatisticalModelsExperiment(BaseExperiment):

    def run(self, *args, **kwargs) -> None:
        # Create readme
        self.create_readme()

        raise NotImplementedError("")


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

    # Features

    FEATURES = [
        "year",
        # "Fertility rate, total",
        # "Population, total",
        # "Net migration",
        # "Arable land",
        # "Birth rate, crude",
        # "GDP growth",
        # "Death rate, crude",
        # "Agricultural land",
        # "Rural population",
        # "Rural population growth",
        # "Age dependency ratio",
        # "Urban population",
        # "Population growth",
        # "Adolescent fertility rate",
        # "Life expectancy at birth, total",
    ]
    # Run experiments
    exp = OptimalParamsExperiment(
        name="OptimalParamsExperiment",
        description="The goal is to find the optimal parameters for the given LocalModel model.",
        hidden_size_range=(128, 2048),
        sequence_length_range=(10, 15),
        num_layers_range=(1, 5),
        learning_rate_range=(1e-5, 1e-2),
    )
    exp.run(state="Czechia", split_rate=0.8, features=FEATURES)

    # Run
    # exp.run(state="Czechia", split_rate=0.8)
