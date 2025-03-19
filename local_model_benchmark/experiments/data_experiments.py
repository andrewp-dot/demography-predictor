"""
In this file are experiments with local model.
"""

# Standard libraries
import os
import pandas as pd
import logging
import pprint
from abc import abstractmethod

from typing import List, Dict
from src.utils.log import setup_logging
from local_model_benchmark.config import LocalModelBenchmarkSettings
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from local_model_benchmark.experiments.base_experiment import BaseExperiment, Experiment

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel


settings = LocalModelBenchmarkSettings()
logger = logging.getLogger("benchmark")

# TODO: make this robust for other architectures -> You need train function, data preprocessing function?
# TODO: FIX all TEMPORARY FIX marks

# Here define experiment types


# Data based experiments
class OneStateDataExperiment(BaseExperiment):

    def run(self, state: str, split_rate: float):
        """
        Trains and evaluates model using a single state data.

        Args:
            state (str): State which data will be used to train model.
            split_rate (float): Split rate for training and validation data.
        """

        # Create readme
        self.create_readme()

        # Define experiment settings
        STATE = state
        state_loader = StateDataLoader(STATE)

        # Single state dataframe
        state_df = state_loader.load_data()

        # Exclude country name
        state_df = state_df.drop(columns=["country name"])

        # Get the model data
        single_state_params = self.model.hyperparameters
        single_state_rnn = self.model

        # Add params to readme
        self.readme_add_params()

        # Add list of features
        self.readme_add_features()

        # Split data
        state_train, state_test = state_loader.split_data(
            state_df, split_rate=split_rate
        )

        # Preproces data
        train_batches, target_batches, state_scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=state_train,
                hyperparameters=single_state_params,
                features=self.FEATURES,
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
            features=self.FEATURES,
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


class AllStatesDataExperiments(BaseExperiment):

    def run(self, state: str, split_rate: float):
        """
        Use whole dataset to train and evaluate model.

        Args:
            state (str): State used for evaluation of the experiment.
            split_rate (float): Split rate for training and validation data.
        """

        # Create readme
        self.create_readme()

        # Load whole dataset
        states_loader = StatesDataLoader()

        all_states = states_loader.load_all_states()

        # Get hyperparameters for training
        all_state_state_params = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )

        # Add params to readme
        self.readme_add_params()

        # Add list of FEATURES
        self.readme_add_features()

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
                states_train_data_dict=states_train_data_dict,
                hyperparameters=all_state_state_params,
                features=self.FEATURES,
            )
        )

        #### Multiple state preprocessing ends here

        # Train rnn
        # all_states_rnn = BaseLSTM(all_state_state_params)

        self.model.train_model(
            batch_inputs=train_input_batches,
            batch_targets=train_target_batches,
            display_nth_epoch=1,
        )

        # Get stats
        stats = self.model.training_stats
        loss_fig = stats.create_plot()
        self.save_plot(fig_name="loss.png", figure=loss_fig)
        self.readme_add_plot(
            plot_name=f"Loss graph", plot_description="", fig_name="loss.png"
        )

        # Evaluate model
        all_states_rnn_evaluation = EvaluateModel(self.model)

        EVAL_STATE = state
        all_states_rnn_evaluation.eval(
            states_train_data_dict[EVAL_STATE][self.FEATURES],
            states_test_data_dict[EVAL_STATE][self.FEATURES],
            features=self.FEATURES,
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


## 1. Use data for just a single state
class Experiment1(Experiment):

    NAME = "OneState"
    DESCRIPTION = "Train and evaluate model on single state data."

    def __init__(self):

        # Define features and parameters and model
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
        self.hyperparameters = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )

        self.model = BaseLSTM(hyperparameters=self.hyperparameters)

        # Define the experiment
        self.exp = OneStateDataExperiment(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
        )

    def run(self, state: str = "Czechia", split_rate: float = 0.8):
        self.exp.run(state=state, split_rate=split_rate)


## 2. Use data for all states (whole dataset)
class Experiment2(Experiment):

    NAME = "AllStates"
    DESCRIPTION = "Train and evaluate model on whole dataset."

    # Define features and parameters and model
    def __init__(self):

        # Define features and parameters and model
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

        self.hyperparameters = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=2048,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=4,
        )

        self.model = BaseLSTM(hyperparameters=self.hyperparameters)

        # Define the experiment
        self.exp = AllStatesDataExperiments(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
        )

    def run(self, state: str = "Czechia", split_rate: float = 0.8):
        self.exp.run(state=state, split_rate=split_rate)


## 2. Use data for all states (whole dataset) without high error features
class Experiment2_1(Experiment):

    NAME = "AllStatesWithoutHighErrorFeatures"
    DESCRIPTION = "Train and evaluate model on whole dataset excluding the high errorneous features."

    def __init__(self):

        # Copy model from Experiment2
        exp2 = Experiment2()

        exclude_features = [
            "population, total",
            "net migration",
        ]
        self.FEATURES = [
            feature for feature in exp2.FEATURES if feature not in exclude_features
        ]

        # Change the input length from exp2 hyperparameters
        exp2_hyperparams = exp2.hyperparameters

        self.hyperparameters = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=exp2_hyperparams.hidden_size,
            sequence_length=exp2_hyperparams.sequence_length,
            learning_rate=exp2_hyperparams.learning_rate,
            epochs=exp2_hyperparams.epochs,
            batch_size=exp2_hyperparams.batch_size,
            num_layers=exp2_hyperparams.num_layers,
        )

        self.model = BaseLSTM(hyperparameters=self.hyperparameters)

        self.exp = AllStatesDataExperiments(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
        )

    def run(self, state: str = "Czechia", split_rate: float = 0.8):
        self.exp.run(state=state, split_rate=split_rate)


## 3. Use data with categories (divide states to categories by GDP in the last year, by geolocation, ...)

## 4. Devide data for aligned sequences (% values - 0 - 100) and for absolute values, which can rise (population, total, ...)


class Experiment3(Experiment):

    NAME = "OnlyStationaryFeatures"
    DESCRIPTION = (
        "Train and evaluate model on single state data using only stationary features."
    )

    def __init__(self):

        # Define the name of the experime

        # Define features and parameters and model
        self.FEATURES = [
            col.lower()
            for col in [
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
        ]
        single_state_params = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )

        self.model = BaseLSTM(hyperparameters=single_state_params)

        # Define the experiment
        self.exp = OneStateDataExperiment(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
        )

    def run(self, state: str = "Czechia", split_rate: float = 0.8):
        self.exp.run(state=state, split_rate=split_rate)


class Experiment3_1(Experiment):

    NAME = "OnlyStationaryFeaturesAllData"
    DESCRIPTION = (
        "Train and evaluate model on whole dataset and using stationary features."
    )

    def __init__(self):

        # Define features and parameters and model
        self.FEATURES = [
            col.lower()
            for col in [
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
        ]
        single_state_params = LSTMHyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=128,
            sequence_length=10,
            learning_rate=0.0001,
            epochs=10,
            batch_size=1,
            num_layers=3,
        )

        self.model = BaseLSTM(hyperparameters=single_state_params)

        # Define the experiment
        self.exp = AllStatesDataExperiments(
            model=self.model,
            name="OnlyStationaryFeaturesAllData",
            description="Train and evaluate model on single state data.",
            features=self.FEATURES,
        )

    def run(self, state: str = "Czechia", split_rate: float = 0.8):
        self.exp.run(state=state, split_rate=split_rate)


# 5. Finetune experiment -> try to use all data from whole dataset and finetune finetunable layers to one state


def run_data_experiments(exp_keys: List[float] = None) -> None:
    """
    Runs all implemented data experiments if not specified the number of experiment. Experiment starts with 1.0.

    Args:
        exp_num (List[int], optional): Number of experiments to run. Defaults to None.
    """

    # Setup experiments
    experiments: Dict[float, Experiment] = {
        1: Experiment1(),
        2: Experiment2(),
        2.1: Experiment2_1(),
        3: Experiment3(),
        3.1: Experiment3_1(),
    }

    # If there are defined experiments keys to run
    if exp_keys is not None:
        for key in exp_keys:
            try:
                logger.info(f"Running experiment {key}...")
                experiments[key].run()
            except KeyError:
                logger.error(
                    f"There is no defined experiment with key: {key}. Available experiment keys: {experiments.keys()}"
                )
        return

    # Else run all experiments
    for key, experiment in experiments.items():
        logger.info(f"Running experiment {key}...")
        experiment.run()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    # run_data_experiments(exp_keys=[1])
    run_data_experiments(exp_keys=[3.1])
