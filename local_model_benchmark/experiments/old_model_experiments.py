# Standard libraries
import os
import pandas as pd
import pprint
import logging
from typing import List, Dict, Literal, Tuple, Union

import copy
import optuna


from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import get_experiment_model, save_experiment_model

from local_model_benchmark.config import get_core_parameters
from local_model_benchmark.experiments.base_experiment import BaseExperiment, Experiment

from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.statistical_models import LocalARIMA
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import RNNHyperparameters, BaseLSTM


# Setup logger
logger = logging.getLogger("benchmark")


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
class OptimalParamsExperiment(BaseExperiment):

    # TODO: evaluation -> use more then r2 score?
    def __init__(
        self,
        model: BaseLSTM,
        name: str,
        description: str,
        features: List[str],
        hidden_size_range: Tuple[int, int],
        sequence_length_range: Tuple[int, int],
        num_layers_range: Tuple[int, int],
        learning_rate_range: Tuple[int, int],
    ):
        super().__init__(model, name, description, features)

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
        self, base_parameters: RNNHyperparameters, hidden_size: int
    ) -> RNNHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.hidden_size = hidden_size
        return new_params

    def adjust_sequence_len(
        self, base_parameters: RNNHyperparameters, sequence_length: int
    ) -> RNNHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.sequence_length = sequence_length
        return new_params

    def adjust_num_layers(
        self, base_parameters: RNNHyperparameters, num_layers: int
    ) -> RNNHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.num_layers = num_layers
        return new_params

    def adjust_learning_rate(
        self, base_parameters: RNNHyperparameters, learning_rate: float
    ):
        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.learning_rate = learning_rate
        return new_params

    def find_optimal_hyperparams(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        base_params: RNNHyperparameters,
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
            NEW_HYPERPARAMS = RNNHyperparameters(
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
            train_batches, target_batches, state_scaler = (
                state_loader.preprocess_training_data_batches(
                    train_data_df=train_df,
                    hyperparameters=NEW_HYPERPARAMS,
                    features=features,
                    scaler=MinMaxScaler(),
                )
            )

            # Train model
            rnn = BaseLSTM(base_params, features=self.FEATURES)

            rnn.set_scaler(state_scaler)

            rnn.train_model(batch_inputs=train_batches, batch_targets=target_batches)

            # Evaluate model
            rnn_evaluation = EvaluateModel(rnn)
            rnn_evaluation.eval(
                test_X=train_df,
                test_y=test_df,
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

    def run(self, state: str, split_rate: float):

        self.create_readme()

        # Load data
        STATE = state
        state_loader = StateDataLoader(STATE)

        state_df = state_loader.load_data()

        # Drop country name
        state_df.drop(columns=["country name"], inplace=True)

        # Get features
        FEATURES = self.FEATURES

        # Split data
        train_data_df, test_data_df = state_loader.split_data(
            state_df, split_rate=split_rate
        )

        best_params_dict = self.find_optimal_hyperparams(
            train_df=train_data_df,
            test_df=test_data_df,
            base_params=self.model.hyperparameters,
            state_loader=state_loader,
            features=FEATURES,
        )

        # Train and evaluate base model
        base_train_batches, base_target_batches, base_scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_data_df,
                hyperparameters=self.model.hyperparameters,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        self.model.set_scaler(base_scaler)

        self.model.train_model(
            batch_inputs=base_train_batches,
            batch_targets=base_target_batches,
            display_nth_epoch=2,
        )

        base_model_evaluation = EvaluateModel(self.model)
        base_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
        )

        # Plot and save base model plot
        base_fig = base_model_evaluation.plot_predictions()

        # Add params
        self.readme_add_params()

        self.readme_add_section(
            title="# Base model evaluation",
            text=f"Hyperparameters:\n```{str(self.model.hyperparameters)}```",
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
            self.model.hyperparameters, best_params_dict["hidden_size"]
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
            state_loader.preprocess_training_data_batches(
                train_data_df=train_data_df,
                hyperparameters=OPTIMAL_PAREMETRS,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        # Train and evaluate the adjusted parameters model
        optimal_model = BaseLSTM(hyperparameters=OPTIMAL_PAREMETRS, features=FEATURES)

        optimal_model.set_scaler(optimal_scaler)

        optimal_model.train_model(
            batch_inputs=optimal_train_batches,
            batch_targets=optimal_target_batches,
            display_nth_epoch=2,
        )

        optimal_model_evaluation = EvaluateModel(optimal_model)
        optimal_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
        )

        # Plot and save base model plot
        optimal_fig = optimal_model_evaluation.plot_predictions()

        self.save_plot(fig_name="optimal_model_eval.png", figure=optimal_fig)

        # Write optimal model evaluation
        self.readme_add_section(
            title="# Optimal model evaluation",
            text="",
        )
        self.readme_add_params(custom_params=optimal_model.hyperparameters)

        self.readme_add_plot(
            plot_name="Optimal model predicted vs reference values",
            plot_description="Displays the performance for every feature predicted of the `Optimal Model`.",
            fig_name="optimal_model_eval.png",
        )

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
