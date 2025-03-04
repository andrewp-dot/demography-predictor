# Standard libraries
import os
import pandas as pd
import torch
from typing import List, Tuple, Union


from config import Config
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from matplotlib.figure import Figure

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters


settings = Config()


# TODO: use this as data preprocessing function
def preprocess_single_state_data(
    train_data_df: pd.DataFrame,
    state_loader: StateDataLoader,
    hyperparameters: LSTMHyperparameters,
    features: List[str],
    scaler: Union[MinMaxScaler, RobustScaler, StandardScaler],
) -> Tuple[
    torch.Tensor, torch.Tensor, Union[MinMaxScaler, RobustScaler, StandardScaler]
]:
    """
    Converts training data to format for training. From the

    Args:
        train_data_df (pd.DataFrame): Unscaled training data.
        state_loader (StateDataLoader): Loader for the state.
        hyperparameters (LSTMHyperparameters): Hyperparameters used for training the model.
        features (List[str]): Features used in model.

    Returns:
        out: Tuple[torch.Tensor, torch.Tensor, Union[MinMaxScaler, RobustScaler, StandardScaler]]: train input batches, train target batches,
        fitted scaler used for training data scaling.
    """

    # Get features
    FEATURES = features

    # Scale data
    scaled_train_data, state_scaler = state_loader.scale_data(
        train_data_df, features=FEATURES, scaler=scaler
    )

    # Create input and target sequences
    train_input_sequences, train_target_sequences = (
        state_loader.preprocess_training_data(
            data=scaled_train_data,
            sequence_len=hyperparameters.sequence_length,
            features=FEATURES,
        )
    )

    # Create input and target batches for faster training
    train_input_batches, train_target_batches = state_loader.create_batches(
        batch_size=hyperparameters.batch_size,
        input_sequences=train_input_sequences,
        target_sequences=train_target_sequences,
    )

    # Return training batches, target batches and fitted scaler
    return train_input_batches, train_target_batches, state_scaler
