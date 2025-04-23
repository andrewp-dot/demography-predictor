# Standard libraries
import os
import pandas as pd
import torch
from torch import nn

from typing import List, Tuple, Union, Dict, Optional, Type

import copy

from config import Config
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Custom imports
from src.utils.save_model import get_experiment_model
from src.base import TrainingStats

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.local_model.model import RNNHyperparameters, BaseRNN
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel
from src.local_model.statistical_models import LocalARIMA

from src.pipeline import LocalModelPipeline


settings = Config()


def preprocess_data(
    data: Dict[str, pd.DataFrame],
    hyperparameters: RNNHyperparameters,
    features: List[str],
    transformer: DataTransformer,
    split_rate: float = 0.8,
    is_fitted: bool = False,
    targets: List[str] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create copy of the data to prevent the inplace modification
    original_data = data.copy()

    # States loader for preprocessing
    states_loader = StatesDataLoader()

    # If the scaler is not fitted, scale and fit data
    if not is_fitted:
        # Get training data and validation data
        train_data_dict, test_data_dict = states_loader.split_data(
            states_dict=data,
            sequence_len=hyperparameters.sequence_length,
            split_rate=split_rate,
        )
        train_states_df = states_loader.merge_states(train_data_dict)
        test_states_df = states_loader.merge_states(test_data_dict)

        scaled_train_data, scaled_validation_data = transformer.scale_and_fit(
            training_data=train_states_df,
            validation_data=test_states_df,
            features=features,
            targets=targets,
        )

        # Create a dictionary from the scaled data
        scaled_data = pd.concat([scaled_train_data, scaled_validation_data], axis=0)
    else:
        original_data = states_loader.merge_states(original_data)
        scaled_data = transformer.scale_data(data=original_data, features=features)

    scaled_states_dict = states_loader.parse_states(scaled_data)

    return transformer.create_train_test_multiple_states_batches(
        data=scaled_states_dict,
        hyperparameters=hyperparameters,
        features=features,
        split_rate=split_rate,
    )


def train_base_rnn(
    name: str,
    hyperparameters: RNNHyperparameters,
    data: Dict[str, pd.DataFrame],
    features: List[str],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
    rnn_type: Optional[Type[Union[nn.LSTM, nn.GRU, nn.RNN]]] = nn.LSTM,
    additional_bpnn: Optional[List[int]] = None,
) -> LocalModelPipeline:

    # Preprocess data
    transformer = DataTransformer()

    batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets = (
        preprocess_data(
            data=data,
            hyperparameters=hyperparameters,
            features=features,
            split_rate=split_rate,
            transformer=transformer,
            is_fitted=False,
        )
    )

    # Create model
    rnn = BaseRNN(
        hyperparameters=hyperparameters,
        features=features,
        rnn_type=rnn_type,
        additional_bpnn=additional_bpnn,
    )

    # Train model
    stats = rnn.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=display_nth_epoch,
    )

    # Create pipeline
    return LocalModelPipeline(
        name=name,
        model=rnn,
        transformer=transformer,
        training_stats=TrainingStats.from_dict(stats),
    )


def train_finetunable_model(
    name: str,
    base_model_pipeline: LocalModelPipeline,
    finetunable_model_hyperparameters: RNNHyperparameters,
    finetunable_model_data: Dict[str, pd.DataFrame],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
    # rnn_type: Optional[Type[Union[nn.LSTM, nn.GRU, nn.RNN]]] = nn.LSTM,
    # additional_bpnn: Optional[List[int]] = None,
) -> LocalModelPipeline:

    # Preprocess data
    batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets = (
        preprocess_data(
            data=finetunable_model_data,
            hyperparameters=finetunable_model_hyperparameters,
            features=base_model_pipeline.model.FEATURES,
            transformer=base_model_pipeline.transformer,
            split_rate=split_rate,
            is_fitted=True,
        )
    )

    # Create model
    finetuneable_lstm = FineTunableLSTM(
        base_model=base_model_pipeline.model,
        hyperparameters=finetunable_model_hyperparameters,
        # rnn_type: Optional[Type[Union[nn.LSTM, nn.GRU, nn.RNN]]] = nn.LSTM,
        # additional_bpnn: Optional[List[int]] = None,
    )

    stats = finetuneable_lstm.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=display_nth_epoch,
    )

    return LocalModelPipeline(
        name=name,
        model=finetuneable_lstm,
        transformer=base_model_pipeline.transformer,
        training_stats=TrainingStats.from_dict(stats),
    )


# TODO: fix this
def train_finetunable_model_from_scratch(
    name: str,
    base_model_hyperparameters: RNNHyperparameters,
    finetunable_model_hyperparameters: RNNHyperparameters,
    base_model_data: Dict[str, pd.DataFrame],
    finetunable_model_data: Dict[str, pd.DataFrame],
    features: List[str],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
) -> LocalModelPipeline:

    # Train base lstm
    base_model_pipeline = train_base_rnn(
        name=name,
        hyperparameters=base_model_hyperparameters,
        data=base_model_data,
        features=features,
        split_rate=split_rate,
        display_nth_epoch=display_nth_epoch,
    )

    return train_finetunable_model(
        name=name,
        base_model_pipeline=base_model_pipeline,
        finetunable_model_hyperparameters=finetunable_model_hyperparameters,
        finetunable_model_data=finetunable_model_data,
        split_rate=split_rate,
        display_nth_epoch=display_nth_epoch,
    )


# TODO: fix this -> make pure ensemble model compatibile for pipeline
def train_ensemble_model(
    name: str,
    hyperparameters: RNNHyperparameters,
    data: Dict[str, pd.DataFrame],
    features: List[str],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
) -> LocalModelPipeline:

    # Ensure the input / output size will be 1
    ADJUSTED_PARAMS = copy.deepcopy(hyperparameters)
    ADJUSTED_PARAMS.input_size = 1  # Predict 1 target at the time
    ADJUSTED_PARAMS.output_size = 1

    # Preprocess data
    transformer = DataTransformer()

    (
        batch_inputs,
        batch_targets,
        batch_validation_inputs,
        batch_validation_targets,
    ) = preprocess_data(
        data=data,
        hyperparameters=ADJUSTED_PARAMS,
        features=features,
        split_rate=split_rate,
        transformer=transformer,
        is_fitted=False,
    )

    trained_models: Dict[str, BaseRNN] = {}
    for i, feature in enumerate(features):

        # Create model
        base_lstm = BaseRNN(hyperparameters=ADJUSTED_PARAMS, features=[feature])

        # Select only the i-th feature for this model
        feature_batch_inputs = batch_inputs[
            :, :, :, i : i + 1
        ]  # (num_samples, batch_size, seq_len, 1)
        feature_batch_targets = batch_targets[
            :, :, :, i : i + 1
        ]  # Same and for validation it is the same

        feature_batch_validation_inputs = batch_validation_inputs[:, :, :, i : i + 1]
        feature_batch_validation_targets = batch_validation_targets[:, :, :, i : i + 1]

        # Train model
        stats = base_lstm.train_model(
            batch_inputs=feature_batch_inputs,
            batch_targets=feature_batch_targets,
            batch_validation_inputs=feature_batch_validation_inputs,
            batch_validation_targets=feature_batch_validation_targets,
            display_nth_epoch=display_nth_epoch,
        )

        trained_models[feature] = base_lstm

    # Create pipeline
    return LocalModelPipeline(
        name=name,
        transformer=transformer,
        model=PureEnsembleModel(feature_models=trained_models),
    )


def train_arima_ensemble_model(
    name: str, features: List[str], state: str, split_rate: float = 0.8
) -> LocalModelPipeline:

    # Load state data
    state_loader = StateDataLoader(state=state)
    state_data = state_loader.load_data()

    # Split data
    train_df, _ = state_loader.split_data(data=state_data, split_rate=split_rate)

    # Train and save models
    trained_models: Dict[str, LocalARIMA] = {}
    for feature in features:

        # Create ARIMA
        arima = LocalARIMA(p=1, d=1, q=1, features=[], target=feature, index="year")

        # Train model
        arima.train_model(data=train_df)

        # Save trained model
        trained_models[feature] = arima

    # Create pipeline
    return LocalModelPipeline(
        name=name,
        transformer=DataTransformer(),
        model=PureEnsembleModel(feature_models=trained_models),
    )
