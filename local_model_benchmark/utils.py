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
        train_data_df, scaler=scaler
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


def write_experiment_results(
    experiment_name: str,
    state: str,
    per_target_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    fig: Figure | None = None,
) -> None:
    """
    Writes experiment model evaluation to file

    Args:
        experiment_name (str): Name of the experiment. This is the folder name of
        state (str): Experiments evaluation results for state (state name)
        per_target_metrics (pd.Dataframe): Metrics dataframe per target.
        overall_metrics (pd.DataFrame): Overall evaluation metrics for prediction.
    """

    experiment_results_dir: str = settings.benchmark_results_dir

    # Create folder if not created
    if not os.path.isdir(experiment_results_dir):
        os.makedirs(experiment_results_dir)

    # Create folder for experiment results
    experiment_dir: str = os.path.join(experiment_results_dir, experiment_name)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    # Save the results
    per_target_metrics.to_csv(
        os.path.join(experiment_dir, f"per_target_metrics_{state}.csv"), index=False
    )
    overall_metrics.to_csv(
        os.path.join(experiment_dir, f"overall_metrics_{state}.csv"), index=False
    )

    # Save figure if given
    if fig is not None:
        fig.savefig(os.path.join(experiment_dir, f"experiment_figure_{state}.png"))
