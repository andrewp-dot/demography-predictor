# Standard library imports
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Type

import torch
from torch import nn

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Custom imports
from src.utils.log import setup_logging
from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.base import RNNHyperparameters
from src.pipeline import GlobalModelPipeline
from src.global_model.model import GlobalModel, XGBoostTuneParams
from src.global_model.global_rnn import GlobalModelRNN


def train_global_model_tree(
    name: str,
    states_data: Dict[str, pd.DataFrame],
    features: List[str],
    targets: List[str],
    sequence_len: int,
    tune_parameters: XGBoostTuneParams,
    tree_model: Union[XGBRegressor, RandomForestRegressor, LGBMRegressor],
    transformer: Optional[DataTransformer] = None,
    split_size: float = 0.8,
) -> GlobalModelPipeline:

    global_model = GlobalModel(
        model=tree_model,
        features=features,
        targets=targets,
        sequence_len=sequence_len,
        params=tune_parameters,
    )

    states_loader = StatesDataLoader()

    # Create train and test for XGB (Global Model)
    X_train, X_test, y_train, _ = global_model.create_train_test_timeseries(
        states_dfs=states_data,
        states_loader=states_loader,
        split_size=split_size,
    )

    # Scale the training data
    if transformer is None:
        transformer = DataTransformer()
        scaled_training_data, _ = transformer.scale_and_fit(
            training_data=X_train,
            validation_data=X_test,
            features=features,
            # targets=targets, # Here are targets useless for tree based model due to transformation of history columns
        )
    else:
        scaled_training_data = transformer.scale_data(data=X_train, features=features)
        # scaled_validation_data = transfomer.scale_data(data=X_test)

    # Train XGB
    global_model.train(
        X_train=scaled_training_data,
        y_train=y_train,
        # tune_hyperparams=True
    )

    # Create Pipeline
    pipeline = GlobalModelPipeline(
        name=name, model=global_model, transformer=transformer
    )
    return pipeline


def preprocess_data_for_rnn(
    states_data: Dict[str, pd.DataFrame],
    hyperparameters: RNNHyperparameters,
    features: List[str],
    targets: List[str],
    transformer: DataTransformer,
    split_rate: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocesses data for the global model RNN.

    Args:
        states_data (Dict[str, pd.DataFrame]): Data used for the model training.
        hyperparameters (RNNHyperparameters): Hyperparameters for the model (which will use these data).
        features (List[str]): Input variables.
        targets (List[str]): Variables to predict.
        split_rate (float): Rate of

    Returns:
        out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets
    """

    states_loader = StatesDataLoader()

    # Get training data and validation data
    train_data_dict, test_data_dict = states_loader.split_data(
        states_dict=states_data,
        sequence_len=hyperparameters.sequence_length,
        split_rate=split_rate,
    )
    train_states_df = states_loader.merge_states(train_data_dict)
    test_states_df = states_loader.merge_states(test_data_dict)

    # Transform data
    scaled_training_data, _ = transformer.scale_and_fit(
        training_data=train_states_df,
        validation_data=test_states_df,
        features=features,
        targets=targets,
    )

    # Create a dictionary from it
    scaled_states_dict = states_loader.parse_states(scaled_training_data)

    batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets = (
        transformer.create_train_test_multiple_states_batches(
            data=scaled_states_dict,
            hyperparameters=hyperparameters,
            features=features + targets,
            targets=targets,
        )
    )

    return (
        batch_inputs,
        batch_targets,
        batch_validation_inputs,
        batch_validation_targets,
    )


def train_global_rnn(
    name: str,
    data: Dict[str, pd.DataFrame],
    hyperparameters: RNNHyperparameters,
    features: List[str],
    targets: List[str],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
    rnn_type: Optional[Type[Union[nn.LSTM, nn.GRU, nn.RNN]]] = nn.LSTM,
) -> GlobalModelPipeline:

    rnn = GlobalModelRNN(hyperparameters, features=features, targets=targets)

    # Create train and test for XGB (Global Model)

    # Preprocess data
    transformer = DataTransformer()

    batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets = (
        preprocess_data_for_rnn(
            states_data=data,
            hyperparameters=hyperparameters,
            features=features,
            targets=targets,
            transformer=transformer,
            split_rate=split_rate,
        )
    )

    # Train RNN
    stats = rnn.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=display_nth_epoch,
    )

    # Create Pipeline
    pipeline = GlobalModelPipeline(name=name, model=rnn, transformer=transformer)
    return pipeline


def main():
    # Load data
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()
    state_df_merged = states_loader.merge_states(state_dfs=state_dfs)

    FEATURES: List[str] = [col.lower() for col in [""]]
    TARGETS: List[str] = [""]

    global_model_pipeline = train_global_model_tree(
        states_data=state_df_merged, features=FEATURES, targets=TARGETS
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run main
    main()
