# Standard library imports
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Type

import torch
from torch import nn
import logging

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import categorical_columns
from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


from src.base import RNNHyperparameters
from src.pipeline import TargetModelPipeline
from src.target_model.model import TargetModelTree, XGBoostTuneParams
from src.target_model.global_rnn import TargetModelRNN

# Import this just for ARIMA
from src.statistical_models.arima import CustomARIMA
from src.feature_model.ensemble_model import PureEnsembleModel
from src.statistical_models.multistate_wrapper import StatisticalMultistateWrapper

logger = logging.getLogger("training")


def train_global_model_tree(
    name: str,
    states_data: Dict[str, pd.DataFrame],
    features: List[str],
    targets: List[str],
    sequence_len: int,
    tree_model: Union[XGBRegressor, RandomForestRegressor, LGBMRegressor],
    xgb_tune_parameters: Optional[XGBoostTuneParams] = None,
    transformer: Optional[DataTransformer] = None,
    split_size: float = 0.8,
) -> TargetModelPipeline:

    global_model = TargetModelTree(
        model=tree_model,
        features=features,
        targets=targets,
        sequence_len=sequence_len,
        params=xgb_tune_parameters,
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

    # Train XGB
    global_model.train(
        X_train=scaled_training_data,
        y_train=y_train,
    )

    # Create Pipeline
    pipeline = TargetModelPipeline(
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
) -> TargetModelPipeline:

    rnn = TargetModelRNN(
        hyperparameters, features=features, targets=targets, rnn_type=rnn_type
    )

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
    pipeline = TargetModelPipeline(name=name, model=rnn, transformer=transformer)
    return pipeline


def train_global_arima_ensemble(
    name: str,
    data: Dict[str, pd.DataFrame],
    features: List[str],
    targets: List[str],
    split_rate: float = 0.8,
    p: int = 1,
    d: int = 1,
    q: int = 1,
) -> TargetModelPipeline:
    """
    Trains CustomARima model

    Args:
        name (str): Name of the pipeline.
        data: (Dict[str, pd.DataFrame]): The training data for the model to train statistical models.
        features (List[str]): Exogeneous features which can influence predictions (need to know their future values).
        targets (List[str]): Targets to predict.
        state (str): State which is used for predictions.
        split_rate (float, optional): The size of the training data. Defaults to 0.8.

    Returns:
        out: TargetModelPipeline: Pipeline with arima ensemble model. Note: transformer in this pipeline is not fitted -> pipeline is created in order to be compatible with comparators etc.
    """

    # Need this loader just to split the state data
    loader = StateDataLoader(state=list(data.keys())[0])
    state_data = loader.load_data()

    transformer = DataTransformer()

    global_states_models: Dict[str, PureEnsembleModel] = {}
    for state, state_data in data.items():

        # Create copy to prevent overwrite of the original list
        arima_model_features = list(features)

        if "year" in features:
            arima_model_features.remove("year")

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

        # Train and save models
        trained_models: Dict[str, CustomARIMA] = {}
        for target in targets:

            # Create ARIMA (or ARMA if d = 0)
            arima = CustomARIMA(
                p=p,
                d=d,
                q=q,
                features=arima_model_features,
                target=target,
                index="year",
                trend="n" if d == 0 else None,
            )

            # Train model
            arima.train_model(data=transformed_train_df)

            # Save trained model
            trained_models[target] = arima

        # Save the pure ensemble for state

        logger.info(f"ARIMA model for state: '{state}' has been created!")
        global_states_models[state] = PureEnsembleModel(
            target_models=trained_models, features=arima_model_features
        )

    model = StatisticalMultistateWrapper(
        model=global_states_models, features=arima_model_features, targets=targets
    )

    # Create pipeline -> put here datatransformer just to in order to create pipeline (for comparators etc)
    return TargetModelPipeline(
        name=name,
        transformer=DataTransformer(),
        model=model,
    )


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
