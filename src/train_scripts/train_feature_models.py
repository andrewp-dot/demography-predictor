# Standard libraries
import pandas as pd
import torch
from torch import nn
import copy


from typing import List, Tuple, Union, Dict, Optional, Type

import pprint

# Custom imports
from config import Config
from src.base import TrainingStats

from src.utils.constants import categorical_columns

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.training_data_transformer import RNNTrainingDataPreprocessor
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.feature_model.model import RNNHyperparameters, BaseRNN
from src.feature_model.finetunable_model import FineTunableLSTM
from src.feature_model.ensemble_model import PureEnsembleModel
from src.statistical_models.arima import CustomARIMA

from src.pipeline import FeatureModelPipeline


from src.statistical_models.multistate_wrapper import StatisticalMultistateWrapper

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

    rnn_preprocessor = RNNTrainingDataPreprocessor()

    # 1. Transform data
    # Transform data for all states
    COLUMNS = features + (targets if targets else [])
    transformed_states_data: Dict[str, pd.DataFrame] = {}
    for state, df in data.items():
        transformed_df = transformer.transform_data(
            state=state, data=df, columns=COLUMNS
        )

        # Check if the data are empty
        if not transformed_df.empty:
            transformed_states_data[state] = transformed_df

    # If the scaler is not fitted, scale and fit data
    if not is_fitted:

        # 2. Split data data
        # Get training data and val idation data
        train_data_dict, test_data_dict = states_loader.split_data(
            states_dict=transformed_states_data,
            sequence_len=hyperparameters.sequence_length,
            split_rate=split_rate,
        )

        # 3a. Scale and fit using training data
        scaled_train_data = transformer.scale_and_fit(
            training_data=states_loader.merge_states(train_data_dict),
            features=features,
            targets=targets,
        )

        # 3b. Scale validation data
        scaled_validation_data_dict: Dict[str, pd.DataFrame] = {}
        for state, df in test_data_dict.items():

            scaled_validation_data_dict[state] = transformer.scale_data(
                data=df, features=features, targets=targets
            )

        scaled_validation_data = states_loader.merge_states(
            state_dfs=scaled_validation_data_dict
        )

        # Merge the data to one scaled dataframe
        scaled_data = pd.concat([scaled_train_data, scaled_validation_data], axis=0)

        scaled_states_dict = states_loader.parse_states(scaled_data)
    else:

        original_data = states_loader.merge_states(transformed_states_data)

        scaled_data_dict: Dict[str, pd.DataFrame] = {}
        for state, df in original_data.items():
            scaled_data_dict[state] = transformer.scale_data(data=df, features=features)

    return rnn_preprocessor.create_train_test_multiple_states_batches(
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
    enable_early_stopping: bool = True,
) -> FeatureModelPipeline:

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

    print(batch_inputs[0][0])
    print(batch_targets[0][0])

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
        enable_early_stopping=enable_early_stopping,
    )

    # Create pipeline
    return FeatureModelPipeline(
        name=name,
        model=rnn,
        transformer=transformer,
        training_stats=TrainingStats.from_dict(stats),
    )


def train_finetunable_model(
    name: str,
    base_model_pipeline: FeatureModelPipeline,
    finetunable_model_hyperparameters: RNNHyperparameters,
    finetunable_model_data: Dict[str, pd.DataFrame],
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
    # rnn_type: Optional[Type[Union[nn.LSTM, nn.GRU, nn.RNN]]] = nn.LSTM,
    # additional_bpnn: Optional[List[int]] = None,
) -> FeatureModelPipeline:

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

    return FeatureModelPipeline(
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
) -> FeatureModelPipeline:

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
) -> FeatureModelPipeline:

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
    return FeatureModelPipeline(
        name=name,
        transformer=transformer,
        model=PureEnsembleModel(target_models=trained_models, features=features),
    )


def train_arima_ensemble_model(
    name: str, targets: List[str], state: str, split_rate: float = 0.8
) -> FeatureModelPipeline:

    # Load state data
    state_loader = StateDataLoader(state=state)
    state_data = state_loader.load_data()

    # Split data
    train_df, _ = state_loader.split_data(data=state_data, split_rate=split_rate)

    # Train and save models
    trained_models: Dict[str, CustomARIMA] = {}
    for target in targets:

        # Create ARIMA
        arima = CustomARIMA(p=1, d=1, q=1, features=[], target=target, index="year")

        # Train model
        arima.train_model(data=train_df)

        # Save trained model
        trained_models[target] = arima

    # Create pipeline
    return FeatureModelPipeline(
        name=name,
        transformer=DataTransformer(),
        model=PureEnsembleModel(target_models=trained_models),
    )


def train_arima_ensemble_all_states(
    name: str,
    data: Dict[str, pd.DataFrame],
    features: List[str],
    split_rate: float = 0.8,
    min_required_points: int = 10,
    p: int = 1,
    d: int = 1,
    q: int = 1,
) -> FeatureModelPipeline:
    """
    Trains CustomARIMA model

    Args:
        name (str): Name of the pipeline.
        data: (Dict[str, pd.DataFrame]): The training data for the model to train statistical models.
        features (List[str]): The features you want to predict. Not the exogeneous variables, but in this case, the features are targets.
        state (str): State which is used for predictions.
        split_rate (float, optional): The size of the training data. Defaults to 0.8.

    Returns:
        out: TargetModelPipeline: Pipeline with arima ensemble model. Note: transformer in this pipeline is not fitted -> pipeline is created in order to be compatible with comparators etc.
    """

    # Need this loader just to split the state data
    loader = StateDataLoader(state=list(data.keys())[0])
    state_data = loader.load_data()

    skipped_states: List[str] = []
    states_models: Dict[str, PureEnsembleModel] = {}

    # Need this to transform data
    transformer = DataTransformer()

    for state, state_data in data.items():

        # Set the exogeneous variables to None.
        arima_model_features = []

        if not "year" in state_data.columns:
            raise ValueError("The input data for ARIMA needs 'year' column!")

        # Split data
        train_df, _ = loader.split_data(data=state_data, split_rate=split_rate)

        # Transform data
        columns = list(train_df.columns)

        # Remove categorical columns
        for categorical_col in categorical_columns():
            columns.remove(categorical_col)

        transformed_train_df = transformer.transform_data(
            data=train_df, columns=columns
        )

        if len(train_df) < min_required_points:  # e.g., 10
            skipped_states.append(state)
            continue

        # Train and save models
        trained_models: Dict[str, CustomARIMA] = {}
        for target in features:

            # Create ARIMA (or ARMA if d = 0)
            arima = CustomARIMA(
                p=p,
                d=d,
                q=q,
                features=arima_model_features,
                target=target,
                index="year",
            )

            # Train model
            arima.train_model(data=transformed_train_df)

            # Save trained model
            trained_models[target] = arima

        # Save the pure ensemble for state

        states_models[state] = PureEnsembleModel(
            target_models=trained_models, features=arima_model_features
        )

    model = StatisticalMultistateWrapper(
        model=states_models, features=arima_model_features, targets=features
    )

    # Create pipeline -> put here datatransformer just to in order to create pipeline (for comparators etc)
    return FeatureModelPipeline(
        name=name,
        transformer=transformer,
        model=model,
    )
