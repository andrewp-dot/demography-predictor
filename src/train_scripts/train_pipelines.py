# Standard library imports
import pandas as pd
from typing import List, Tuple

# Custom library imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.base import LSTMHyperparameters

from src.global_model.model import XGBoostTuneParams

from src.pipeline import PredictorPipeline

from src.train_scripts.train_local_models import train_base_lstm
from src.train_scripts.train_global_models import train_global_model

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

# # Technically I do not have to scale the both -> maybe I will just use 2 pipelines in the row and celebrate that I do not have to do this shit
# # Problems in here: the local model and global model would be trained on the exact same training (time series) data

# def preprocess_data_for_pipeline_training(data: pd.DataFrame, local_model_features: List[str], additional_global_model_features: List[str] = None,) -> Tuple[pd.DataFrame, DataTransformer]:
#     # Create transformer
#     transformer = DataTransformer()
#     if additional_global_model_features is None:
#         additional_global_model_features = []

#     # Get all features to scale


def train_basic_pipeline(
    hyperparameters: LSTMHyperparameters,
    local_model_features: List[str],
    global_model_targets: List[str],
    additional_global_model_features: List[str] = None,
    split_rate: float = 0.8,
    display_nth_epoch=10,
) -> PredictorPipeline:

    # Load whole dataset
    loader = StatesDataLoader()
    all_states_data_dict = loader.load_all_states()
    merged_all_states_data = loader.merge_states(state_dfs=all_states_data_dict)

    # Train local model pipeline
    local_model_pipeline = train_base_lstm(
        hyperparameters=hyperparameters,
        data=all_states_data_dict,
        features=local_model_features,
        split_rate=split_rate,
        display_nth_epoch=display_nth_epoch,
    )

    # Train global model pieplien
    tune_parameters = XGBoostTuneParams(
        n_estimators=[50, 100],
        learning_rate=[0.001, 0.01, 0.05],
        max_depth=[3, 5, 7],
        subsample=[0.5, 0.7],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    global_model_pipeline = train_global_model(
        data=merged_all_states_data,
        features=[*local_model_features, *additional_global_model_features],
        targets=global_model_targets,
        tune_parameters=tune_parameters,
    )

    # Create pipeline
    model_pipeline = PredictorPipeline(
        local_model_pipeline=local_model_pipeline,
        global_model_pipeline=global_model_pipeline,
    )

    return model_pipeline


def main():

    # Get local model features
    LOCAL_MODEL_FEATURES: List[str] = [col.lower() for col in [""]]
    GLOBAL_MODEL_ADDITIONAL_FEATURES: List[str] = [
        col.lower() for col in ["year", "country name"]
    ]

    hyperparameters = get_core_hyperparameters(input_size=len(LOCAL_MODEL_FEATURES))

    train_basic_pipeline(hyperparameters=hyperparameters)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run main
    main()

    raise NotImplementedError("Running this is not implemented yet!")
