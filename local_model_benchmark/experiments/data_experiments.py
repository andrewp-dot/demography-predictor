"""
In this file are experiments with local model.
"""

# Standard libraries
import os
import pandas as pd
import logging
import pprint

from typing import List
from config import setup_logging, Config
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from local_model_benchmark.experiments.base_experiment import BaseExperiment

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateModel


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: make this robust for other architectures -> You need train function, data preprocessing function?
# TODO: FIX all TEMPORARY FIX marks

# Get the list of all available features
ALL_FEATURES = [
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

ALL_FEATURES = [col.lower() for col in ALL_FEATURES]


# Setup features to use all
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

FEATURES = [col.lower() for col in FEATURES]


# Data based experiments
## 1. Use data for just a single state


class OneStateDataExperiment(BaseExperiment):

    def run(self, state: str, split_rate: float, features: List[str]):
        """
        Trains and evaluates model using a single state data.

        Args:
            state (str): State which data will be used to train model.
            split_rate (float): Split rate for training and validation data.
        """

        # Get features
        FEATURES = features

        # Create readme
        self.create_readme()

        # Define experiment settings
        STATE = state
        state_loader = StateDataLoader(STATE)

        # Single state dataframe
        state_df = state_loader.load_data()

        # Exclude country name
        state_df = state_df.drop(columns=["country name"])

        single_state_params = LSTMHyperparameters(
            input_size=len(FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )
        single_state_rnn = LocalModel(single_state_params)

        # Add params to readme
        self.readme_add_section(
            title="## Hyperparameters", text=f"```{single_state_params}```"
        )

        # Add list of features
        self.readme_add_section(
            title="## Features", text="```\n" + "\n".join(FEATURES) + "\n```"
        )

        # Split data
        state_train, state_test = state_loader.split_data(
            state_df, split_rate=split_rate
        )

        # Preproces data
        train_batches, target_batches, state_scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=state_train,
                hyperparameters=single_state_params,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        # Train model
        single_state_rnn.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=1,
        )

        # Get training stats
        stats = single_state_rnn.training_stats
        fig = stats.create_plot()

        self.save_plot(fig_name="loss.png", figure=fig)
        self.readme_add_plot(
            plot_name=f"Loss graph", plot_description="", fig_name="loss.png"
        )

        # Evaluate model
        single_state_rnn_evaluation = EvaluateModel(single_state_rnn)
        single_state_rnn_evaluation.eval(
            state_train,
            state_test,
            features=FEATURES,
            scaler=state_scaler,
        )

        # Get figure
        fig = single_state_rnn_evaluation.plot_predictions()

        self.save_plot(fig_name="evaluation.png", figure=fig)
        self.readme_add_plot(
            plot_name=f"Prediction of {state} by the training data",
            plot_description="",
            fig_name="evaluation.png",
        )

        # Save the results
        formatted_model_evaluation: str = pprint.pformat(
            single_state_rnn_evaluation.to_readable_dict()
        )

        self.readme_add_section(
            title="# Metric result", text=formatted_model_evaluation
        )


## 2. Use data for all states (whole dataset)
class AllStatesDataExperiments(BaseExperiment):

    def run(self, state: str, split_rate: float, features: List[str]):
        """
        Use whole dataset to train and evaluate model.

        Args:
            state (str): State used for evaluation of the experiment.
            split_rate (float): Split rate for training and validation data.
        """

        # Set features const
        FEATURES = features

        # Create readme
        self.create_readme()

        # Load whole dataset
        states_loader = StatesDataLoader()

        all_states = states_loader.load_all_states()

        # Get hyperparameters for training
        all_state_state_params = LSTMHyperparameters(
            input_size=len(FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )

        # Add params to readme
        self.readme_add_section(
            title="## Hyperparameters", text=f"```{all_state_state_params}```"
        )

        # Add list of features
        self.readme_add_section(
            title="## Features", text="```\n" + "\n".join(FEATURES) + "\n```"
        )

        # TODO: Maybe you an write this to all in one function

        #### Multiple state preprocessing starts here

        # Split data
        states_train_data_dict, states_test_data_dict = states_loader.split_data(
            states_dict=all_states,
            sequence_len=all_state_state_params.sequence_length,
            split_rate=split_rate,
        )

        # Get train batches, target batches, and fitted scaler
        train_input_batches, train_target_batches, all_states_scaler = (
            states_loader.preprocess_train_data_batches(
                all_states=states_train_data_dict,
                hyperparameters=all_state_state_params,
                features=FEATURES,
                split_rate=0.8,
            )
        )

        #### Multiple state preprocessing ends here

        # Train rnn
        all_states_rnn = LocalModel(all_state_state_params)

        all_states_rnn.train_model(
            batch_inputs=train_input_batches,
            batch_targets=train_target_batches,
            display_nth_epoch=1,
        )

        # Get stats
        stats = all_states_rnn.training_stats
        loss_fig = stats.create_plot()
        self.save_plot(fig_name="loss.png", figure=loss_fig)
        self.readme_add_plot(
            plot_name=f"Loss graph", plot_description="", fig_name="loss.png"
        )

        # Evaluate model
        all_states_rnn_evaluation = EvaluateModel(all_states_rnn)

        EVAL_STATE = state
        all_states_rnn_evaluation.eval(
            states_train_data_dict[EVAL_STATE][FEATURES],
            states_test_data_dict[EVAL_STATE][FEATURES],
            features=FEATURES,
            scaler=all_states_scaler,
        )

        evaluation_fig = all_states_rnn_evaluation.plot_predictions()

        evaluation_fig_name = f"evaluation_{EVAL_STATE}.png".lower()
        self.save_plot(fig_name=evaluation_fig_name, figure=evaluation_fig)

        self.readme_add_plot(
            plot_name=f"Evaluation of the model - state: {EVAL_STATE}",
            plot_description="",
            fig_name=evaluation_fig_name,
        )

        # Save the results
        formatted_model_evaluation: str = pprint.pformat(
            all_states_rnn_evaluation.to_readable_dict()
        )

        self.readme_add_section(
            title="# Metric result", text=formatted_model_evaluation
        )


## 3. Use data with categories (divide states to categories by GDP in the last year, by geolocation, ...)


## 4. Devide data for aligned sequences (% values - 0 - 100) and for absolute values, which can rise (population, total, ...)
class OnlyStationaryFeaturesDataExperiment(BaseExperiment):

    def run(self, state: str, split_rate: float) -> None:
        """
        Trains model using one state data only with stationary features.

        Args:
            state (str): State which data will be used to train model.
            split_rate (float): Split rate for training and validation data.
        """

        # Create reamde
        self.create_readme()

        # Load the state
        STATE = state
        state_loader = StateDataLoader(STATE)
        state_data_df = state_loader.load_data()

        # Get only numerical features
        FEATURES = [
            # Need to run columns
            "year",
            # Stationary columns
            "Fertility rate, total",
            "Arable land",
            "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Population ages 15-64",
            "Population ages 0-14",
            "Agricultural land",
            "Population ages 65 and above",
            "Rural population",
            "Rural population growth",
            # "Age dependency ratio",
            "Urban population",
            "Population growth",
        ]

        # Adjust feature names to lower
        FEATURES = [col.lower() for col in FEATURES]

        # Get only data with features
        state_data_df = state_data_df[FEATURES]

        # Get hyperparameters for training
        only_staionary_data_params = LSTMHyperparameters(
            input_size=len(FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=40,
            batch_size=1,
            num_layers=3,
        )

        # Add params to readme
        self.readme_add_section(
            title="## Hyperparameters", text=f"```{only_staionary_data_params}```"
        )

        # Add list of features
        self.readme_add_section(
            title="## Features", text="```\n" + "\n".join(FEATURES) + "\n```"
        )

        # Split data
        train_data_df, test_data_df = state_loader.split_data(
            data=state_data_df, split_rate=split_rate
        )

        # Preprocess data
        train_input_batches, train_target_batches, state_scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_data_df,
                hyperparameters=only_staionary_data_params,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        # Train rnn
        only_stationary_rnn = LocalModel(only_staionary_data_params)

        only_stationary_rnn.train_model(
            batch_inputs=train_input_batches,
            batch_targets=train_target_batches,
        )

        # Get stats
        stats = only_stationary_rnn.training_stats
        fig = stats.create_plot()

        # Save training stats or plot it

        # Evaluate model
        only_stationary_rnn_evaluation = EvaluateModel(only_stationary_rnn)

        only_stationary_rnn_evaluation.eval(
            train_data_df,
            test_data_df,
            features=FEATURES,
            scaler=state_scaler,
        )

        fig = only_stationary_rnn_evaluation.plot_predictions()
        self.save_plot(fig_name="evaluation.png", figure=fig)
        self.readme_add_plot(
            plot_name=f"Evaluation of the model",
            plot_description="",
            fig_name="evaluation.png",
        )

        # Save the results
        formatted_model_evaluation: str = pprint.pformat(
            only_stationary_rnn_evaluation.to_readable_dict()
        )

        self.readme_add_section(
            title="# Compare metric results", text=formatted_model_evaluation
        )


# 5. Finetune experiment -> try to use all data from whole dataset and finetune finetunable layers to one state


def run_data_experiments() -> None:
    """
    Runs all implemented data experiments
    """

    # TODO: make experiments robust for trying different base parameters

    # Setup experiments
    exp1 = OneStateDataExperiment(
        name="OneStateDataExperiment",
        description="Train and evaluate model on single state data.",
    )
    EXP1_FEATURES = [
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
    EXP1_FEATURES = [col.lower() for col in EXP1_FEATURES]

    exp2 = AllStatesDataExperiments(
        name="AllStatesDataExperiments",
        description="Train and evaluate model on whole dataset.",
    )
    EXP2_FEATURES = [
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
    EXP2_FEATURES = [col.lower() for col in EXP2_FEATURES]

    exp2_1 = AllStatesDataExperiments(
        name="AllStatesDataExperimentsWithoutHighErrorFeatures",
        description="Train and evaluate model on whole dataset.",
    )
    EXP2_1_FEATURES = [
        "year",
        "Fertility rate, total",
        # "Population, total",
        # "Net migration",
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
    EXP2_1_FEATURES = [col.lower() for col in EXP2_1_FEATURES]

    exp3 = OnlyStationaryFeaturesDataExperiment(
        name="OnlyStationaryFeaturesDataExperiment",
        description="Train and evaluate model on single state data with all features with boundaries (for example % values, features with some mean.) ",
    )

    # Run experiments with parameters
    # exp1.run(state="Czechia", split_rate=0.8, features=EXP1_FEATURES)
    # exp2.run(state="Czechia", split_rate=0.8, features=EXP2_FEATURES)
    exp2_1.run(state="Czechia", split_rate=0.8, features=EXP2_1_FEATURES)
    # exp3.run(state="Czechia", split_rate=0.8)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    run_data_experiments()
