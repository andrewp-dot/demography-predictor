"""
In this file are experiments with local model. 
"""

# Standard libraries
import os
import pandas as pd
import logging


from config import setup_logging, Config
from local_model_benchmark.utils import (
    preprocess_single_state_data,
    write_experiment_results,
)
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateLSTM


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: Data preprocessing functions -> split-> scaling -> sequences -> batches
# TODO: doc comments

## Maybe here define base experiment function

# Data based experiments


## 1. Use data for just a single state
def single_state_data_experiment(state: str, split_rate: float) -> None:
    # Define experiment name
    EXPERIMENT_NAME = single_state_data_experiment.__name__

    # Define experiment settings
    STATE = state
    state_loader = StateDataLoader(STATE)

    # Single state dataframe
    state_df = state_loader.load_data()

    # Exclude country name
    state_df = state_df.drop(columns=["country name"])

    # Get features
    FEATURES = [col.lower() for col in state_df.columns]

    single_state_params = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=32,
        sequence_length=10,
        learning_rate=0.001,
        epochs=10,
        batch_size=1,  # Edit this for faster training
        num_layers=4,
    )
    single_state_rnn = LocalModel(single_state_params)

    # Split data
    state_train, state_test = state_loader.split_data(state_df, split_rate=split_rate)

    # Preproces data
    train_batches, target_batches, state_scaler = preprocess_single_state_data(
        train_data_df=state_train,
        state_loader=state_loader,
        hyperparameters=single_state_params,
        features=FEATURES,
        scaler=MinMaxScaler(),
    )

    # Train model
    single_state_rnn.train_model(
        batch_inputs=train_batches, batch_targets=target_batches, display_nth_epoch=1
    )

    # Get stats
    stats = single_state_rnn.training_stats
    fig = stats.create_plot()

    # Save training stats or plot it

    # Evaluate model
    single_state_rnn_evaluation = EvaluateLSTM(single_state_rnn)
    single_state_rnn_evaluation.eval(
        state_train,
        state_test,
        features=FEATURES,
        scaler=state_scaler,
    )

    import matplotlib.pyplot as plt

    prediction_plot = single_state_rnn_evaluation.plot_predictions()
    prediction_plot.savefig("./test_plot")

    # Get evaluation metrics
    write_experiment_results(
        EXPERIMENT_NAME,
        state=STATE,
        per_target_metrics=single_state_rnn_evaluation.per_target_metrics,
        overall_metrics=single_state_rnn_evaluation.overall_metrics,
        fig=fig,
    )


## 2. Use data for all states (whole dataset)
def whole_dataset_experiment() -> None:

    # Define experiment name
    EXPERIMENT_NAME = whole_dataset_experiment.__name__

    # Load whole dataset
    states_loader = StatesDataLoader()

    all_states = states_loader.load_all_states()

    # Get only numerical features
    FEATURES = [
        col.lower()  # Lower to ensure key compatibility
        for col in all_states["Czechia"].select_dtypes(include="number").columns
    ]

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

    # TODO: Maybe you an write this to all in one function

    # Split data
    states_train_data_dict, states_test_data_dict = states_loader.split_data(
        states_dict=all_states, sequence_len=all_state_state_params.sequence_length
    )

    # Scale data
    scaled_train_data, all_states_scaler = states_loader.scale_data(
        states_train_data_dict, scaler=MinMaxScaler()
    )

    # Create input and target sequences
    train_input_sequences, train_target_sequences = (
        states_loader.create_train_sequences(
            states_data=scaled_train_data,
            sequence_len=all_state_state_params.sequence_length,
            features=FEATURES,
        )
    )

    # Create input and target batches for faster training
    train_input_batches, train_target_batches = states_loader.create_train_batches(
        input_sequences=train_input_sequences,
        target_sequences=train_target_sequences,
        batch_size=all_state_state_params.batch_size,
    )

    # Train rnn
    all_states_rnn = LocalModel(all_state_state_params)

    all_states_rnn.train_model(
        batch_inputs=train_input_batches,
        batch_targets=train_target_batches,
        display_nth_epoch=1,
    )

    # Get stats
    stats = all_states_rnn.training_stats
    fig = stats.create_plot()

    # Save training stats or plot it

    # Evaluate model
    all_states_rnn_evaluation = EvaluateLSTM(all_states_rnn)

    EVAL_STATE = "Czechia"
    all_states_rnn_evaluation.eval(
        states_train_data_dict[EVAL_STATE][FEATURES],
        states_test_data_dict[EVAL_STATE][FEATURES],
        features=FEATURES,
        scaler=all_states_scaler,
    )

    # Save the results
    write_experiment_results(
        EXPERIMENT_NAME,
        state=EVAL_STATE,
        per_target_metrics=all_states_rnn_evaluation.per_target_metrics,
        overall_metrics=all_states_rnn_evaluation.overall_metrics,
        fig=fig,
    )


## 3. Use data with categories (divide states to categories by GDP in the last year, by geolocation, ...)


## 4. Devide data for aligned sequences (% values - 0 - 100) and for absolute values, which can rise (population, total, ...)
def only_stationary_data_experiment(state: str, split_rate: float) -> None:

    # Define experiment name
    EXPERIMENT_NAME = only_stationary_data_experiment.__name__

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

    # Split data
    train_data_df, test_data_df = state_loader.split_data(
        data=state_data_df, split_rate=split_rate
    )

    # Preprocess data
    train_input_batches, train_target_batches, state_scaler = (
        preprocess_single_state_data(
            train_data_df=train_data_df,
            state_loader=state_loader,
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
    only_stationary_rnn_evaluation = EvaluateLSTM(only_stationary_rnn)

    only_stationary_rnn_evaluation.eval(
        train_data_df,
        test_data_df,
        features=FEATURES,
        scaler=state_scaler,
    )

    # Save the results
    write_experiment_results(
        EXPERIMENT_NAME,
        state=STATE,
        per_target_metrics=only_stationary_rnn_evaluation.per_target_metrics,
        overall_metrics=only_stationary_rnn_evaluation.overall_metrics,
        fig=fig,
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    single_state_data_experiment(state="Czechia", split_rate=0.8)
    whole_dataset_experiment()
    only_stationary_data_experiment(state="Czechia", split_rate=0.8)
