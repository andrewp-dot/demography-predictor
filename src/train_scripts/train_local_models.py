# Standard libraries
import os
import pandas as pd
import torch
from typing import List, Tuple, Union, Dict, Optional


from config import Config
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Custom imports
from src.utils.save_model import get_experiment_model
from src.base import TrainingStats

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel

from src.pipeline import LocalModelPipeline


settings = Config()


def preprocess_data(
    data: Dict[str, pd.DataFrame],
    hyperparameters: LSTMHyperparameters,
    features: List[str],
    transformer: DataTransformer,
    split_rate: float = 0.8,
    is_fitted: bool = False,
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

        scaled_train_data, scaled_validation_data, _ = transformer.scale_and_fit(
            training_data=train_states_df,
            validation_data=test_states_df,
            features=features,
            scaler=MinMaxScaler(),
        )

        # Create a dictionary from the scaled data
        scaled_data = pd.concat([scaled_train_data, scaled_validation_data], axis=0)
    else:
        scaled_data = transformer.scale_data(data=original_data, columns=features)

    scaled_states_dict = states_loader.parse_states(scaled_data)

    return transformer.create_train_test_multiple_states_batches(
        data=scaled_states_dict,
        hyperparameters=hyperparameters,
        features=features,
        split_rate=split_rate,
    )


def train_base_lstm(
    hyperparameters: LSTMHyperparameters,
    data: Dict[str, pd.DataFrame],
    features: List[str],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
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
    base_lstm = BaseLSTM(hyperparameters=hyperparameters, features=features)

    # Train model
    stats = base_lstm.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=display_nth_epoch,
    )

    # Create pipeline
    return LocalModelPipeline(
        model=base_lstm,
        transformer=transformer,
        training_stats=TrainingStats.from_dict(stats),
    )


def train_finetunable_model(
    base_model_pipeline: LocalModelPipeline,
    finetunable_model_hyperparameters: LSTMHyperparameters,
    finetunable_model_data: Dict[str, pd.DataFrame],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
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
    )

    stats = finetuneable_lstm.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=display_nth_epoch,
    )

    return LocalModelPipeline(
        model=finetuneable_lstm,
        transformer=base_model_pipeline.transformer,
        training_stats=TrainingStats.from_dict(stats),
    )


def train_finetunable_model_from_scratch(
    base_model_hyperparameters: LSTMHyperparameters,
    finetunable_model_hyperparameters: LSTMHyperparameters,
    base_model_data: Dict[str, pd.DataFrame],
    finetunable_model_data: Dict[str, pd.DataFrame],
    features: List[str],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
) -> LocalModelPipeline:

    # Train base lstm
    base_model_pipeline = train_base_lstm(
        hyperparameters=base_model_hyperparameters,
        data=base_model_data,
        features=features,
        split_rate=split_rate,
        display_nth_epoch=display_nth_epoch,
    )

    return train_finetunable_model(
        base_model_pipeline=base_model_pipeline,
        finetunable_model_hyperparameters=finetunable_model_hyperparameters,
        finetunable_model_data=finetunable_model_data,
        split_rate=split_rate,
        display_nth_epoch=display_nth_epoch,
    )
