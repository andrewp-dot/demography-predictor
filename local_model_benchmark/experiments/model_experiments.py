# Standard libraries
import pandas as pd
import pprint
import logging
from typing import List, Dict, Literal, Tuple

import copy
import optuna


from config import Config
from src.utils.log import setup_logging
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from local_model_benchmark.experiments.base_experiment import BaseExperiment, Experiment

from src.local_model.statistical_models import LocalARIMA, EvaluateARIMA
from src.preprocessors.state_preprocessing import StateDataLoader
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel


# Setup logger
logger = logging.getLogger("benchmark")

# Get settings
settings = Config()

# TODO: rework these experiments
# 1. Stash changes
# 2. Add them to CLI

## Model experiments settings

# Get the list of all available features


# Setup features to use all
FEATURES = [
    "year",
    "Fertility rate, total",
    "Population, total",
    "Net migration",
    "Arable land",
    "Birth rate, crude",
    "GDP growth",
    "Death rate, crude",
    "Agricultural land",
    "Rural population",
    "Rural population growth",
    "Age dependency ratio",
    "Urban population",
    "Population growth",
    "Adolescent fertility rate",
    "Life expectancy at birth, total",
]

FEATURES = [col.lower() for col in FEATURES]

BASE_HYPERPARAMS = LSTMHyperparameters(
    input_size=len(FEATURES),
    hidden_size=256,
    sequence_length=10,
    learning_rate=0.0001,
    epochs=20,
    batch_size=1,
    num_layers=3,
)


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
            train_batches, target_batches, state_scaler = (
                state_loader.preprocess_training_data_batches(
                    train_data_df=train_df,
                    hyperparameters=NEW_HYPERPARAMS,
                    features=features,
                    scaler=MinMaxScaler(),
                )
            )

            # Train model
            rnn = BaseLSTM(base_params)
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

        self.model.train_model(
            batch_inputs=base_train_batches,
            batch_targets=base_target_batches,
            display_nth_epoch=2,
        )

        base_model_evaluation = EvaluateModel(self.model)
        base_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
            features=FEATURES,
            scaler=base_scaler,
        )

        # Plot and save base model plot
        base_fig = base_model_evaluation.plot_predictions()

        # Add params
        self.readme_add_params()

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
        optimal_model = BaseLSTM(hyperparameters=OPTIMAL_PAREMETRS)

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


# 2. Compare model with statistical methods (ARIMA, GM)
# 2.1. VAR, SARIMA, ARIMA * 19?
class CompareLSTMARIMAExperiment(BaseExperiment):

    def run(self, state: str, split_rate: float) -> None:
        # Create readme
        self.create_readme()

        # Adjust parameters for RNN and print them into the README.md
        ADJUSTED_PARAMS = copy.deepcopy(BASE_HYPERPARAMS)
        ADJUSTED_PARAMS.input_size = (
            1  # Set the input size to 1, beacause there is only 1 feature
        )

        self.readme_add_section(
            title="# LSTM model parameters",
            text=f"Hyperparameters:\n```{str(BASE_HYPERPARAMS)}```",
        )

        # Load data
        STATE = state
        state_loader = StateDataLoader(STATE)

        state_df = state_loader.load_data()

        # Split data
        train_df, test_df = state_loader.split_data(
            data=state_df, split_rate=split_rate
        )

        # Exclude "year" from features
        features = self.FEATURES
        features.remove("year")

        # For every features create ARIMA and LSTM model
        for feature in features:

            logger.info(f"[{self.name}]: ### Current target: {feature} ###")

            # Set feature as a target
            target = feature

            # Create new section in README.md
            self.readme_add_section(
                title=f"# LSTM & ARIMA Comparision: Feature: {target}",
                text=f"Comparision of LSTM and ARIMA model of predicting feature {target}. State: {state}",
            )

            # Create ARIMA
            arima = LocalARIMA(1, 1, 1, features=[], target=target, index="year")
            arima.train_model(train_df)

            # Evaluate ARIMA
            arima_evaluation = EvaluateARIMA(arima=arima)
            arima_evaluation.eval(
                test_X=train_df,
                test_y=test_df,
                features=[],
                target=target,
                index="year",
            )

            # Save ARIMA evaluation
            arima_predictions_fig = arima_evaluation.plot_predictions()
            arima_fig_name = f"arima_evaluation_{target}.png".replace(" ", "_")

            self.save_plot(fig_name=arima_fig_name, figure=arima_predictions_fig)
            self.readme_add_plot(
                plot_name="Arima evaluation",
                plot_description="",
                fig_name=arima_fig_name,
            )

            # Create RNN
            rnn = BaseLSTM(hyperparameters=ADJUSTED_PARAMS)

            # Preprocess data
            input_batches, target_batches, scaler = (
                state_loader.preprocess_training_data_batches(
                    train_data_df=train_df,
                    hyperparameters=ADJUSTED_PARAMS,
                    features=[target],
                    scaler=MinMaxScaler(),
                )
            )

            # Train model
            rnn.train_model(
                batch_inputs=input_batches,
                batch_targets=target_batches,
                display_nth_epoch=1,
            )

            # Evaluate model
            rnn_evaluation = EvaluateModel(model=rnn)

            rnn_evaluation.eval(
                test_X=train_df, test_y=test_df, features=[target], scaler=scaler
            )

            # Save LSTM evaluation
            rnn_fig = rnn_evaluation.plot_predictions()
            rnn_fig_name = f"lstm_evaluation_{target}.png".replace(" ", "_")

            self.save_plot(fig_name=rnn_fig_name, figure=rnn_fig)
            self.readme_add_plot(
                plot_name=f"RNN evaluation - {target}",
                plot_description="",
                fig_name=rnn_fig_name,
            )

            # Print metrics to file
            formatted_rnn_eval = pprint.pformat(rnn_evaluation.to_readable_dict())
            formatted_arima_eval = pprint.pformat(arima_evaluation.to_readable_dict())

            self.readme_add_section(
                title="### Overall metrics (ARIMA)",
                text=f"```\n{formatted_arima_eval}\n```\n",
            )
            self.readme_add_section(
                title="### Overall metrics (RNN)",
                text=f"```\n{formatted_rnn_eval}\n```\n",
            )


# 3. Compare prediction using whole state data and the last few records of data
# 4. Predict parameters for different years (e.g. to 2030, 2040, ... )


class LSTMOptimalParameters(Experiment):

    NAME = "OptimalParamsExperiment"
    DESCRIPTION = "Compares the performance of LSTM model with the statistical ARIMA model for all features prediction. Each model is trained just to predict 1 feauture from all features."

    def __init__(self):

        self.name = self.NAME

        self.FEATURES = [
            col.lower()
            for col in [
                "year",
                "Fertility rate, total",
                "Population, total",
                "Net migration",
                "Arable land",
                "Birth rate, crude",
                "GDP growth",
                "Death rate, crude",
                "Agricultural land",
                "Rural population",
                "Rural population growth",
                "Age dependency ratio",
                "Urban population",
                "Population growth",
                "Adolescent fertility rate",
                "Life expectancy at birth, total",
            ]
        ]

        hyperparameters = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=256,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=20,
            batch_size=1,
            num_layers=3,
        )

        self.model = BaseLSTM(hyperparameters=hyperparameters)

        self.exp = OptimalParamsExperiment(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
            hidden_size_range=(128, 2048),
            sequence_length_range=(10, 15),
            num_layers_range=(1, 5),
            learning_rate_range=(1e-5, 1e-2),
        )

    def run(self, state="Czechia", split_rate=0.8):
        return self.exp.run(state=state, split_rate=split_rate)


class RNNvsStatisticalMethods(Experiment):
    NAME = "CompareLSTMARIMAExperiment"
    DESCRIPTION = (
        "The goal is to find the optimal parameters for the given BaseLSTM model."
    )

    def __init__(self):

        self.name = self.NAME

        self.FEATURES = [
            col.lower()
            for col in [
                "year",
                "Fertility rate, total",
                "Population, total",
                "Net migration",
                "Arable land",
                "Birth rate, crude",
                "GDP growth",
                "Death rate, crude",
                "Agricultural land",
                "Rural population",
                "Rural population growth",
                "Age dependency ratio",
                "Urban population",
                "Population growth",
                "Adolescent fertility rate",
                "Life expectancy at birth, total",
            ]
        ]

        hyperparameters = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=256,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=20,
            batch_size=1,
            num_layers=3,
        )

        self.model = BaseLSTM(hyperparameters=hyperparameters)

        self.exp = CompareLSTMARIMAExperiment(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
        )

    def run(self, state="Czechia", split_rate=0.8):
        return self.exp.run(state=state, split_rate=split_rate)


def run_experiments():
    exp1 = LSTMOptimalParameters()
    exp1.run()

    exp2 = RNNvsStatisticalMethods()
    exp2.run()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    run_experiments()

    # Run
    # exp.run(state="Czechia", split_rate=0.8)
