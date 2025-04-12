# Standard library imports
import pandas as pd
from typing import List, Tuple, Dict

# Custom library imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.base import LSTMHyperparameters
from src.evaluation import EvaluateModel

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


RICH_STATES: List[str] = [
    "Australia",
    "Austria",
    "Bahamas, The",
    "Bahrain",
    "Belgium",
    "Brunei Darussalam",
    "Canada",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Hong Kong SAR, China",
    "Iceland",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Korea, Rep.",
    "Kuwait",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Oman",
    "Poland",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Singapore",
    "Slovak Republic",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
]


def train_basic_pipeline(
    name: str,
    global_model_data: pd.DataFrame,
    local_model_data_dict: Dict[str, pd.DataFrame],
    hyperparameters: LSTMHyperparameters,
    local_model_features: List[str],
    global_model_targets: List[str],
    additional_global_model_features: List[str] = None,
    split_rate: float = 0.8,
    display_nth_epoch=10,
) -> PredictorPipeline:

    # Train local model pipeline
    local_model_pipeline = train_base_lstm(
        hyperparameters=hyperparameters,
        data=local_model_data_dict,
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
        data=global_model_data,
        features=[*local_model_features, *additional_global_model_features],
        targets=global_model_targets,
        tune_parameters=tune_parameters,
    )

    # Create pipeline
    model_pipeline = PredictorPipeline(
        name=name,
        local_model_pipeline=local_model_pipeline,
        global_model_pipeline=global_model_pipeline,
    )

    return model_pipeline


def main():

    # Get local model features
    LOCAL_MODEL_FEATURES: List[str] = [
        col.lower()
        for col in [
            # "year",
            "Fertility rate, total",
            # "Population, total",
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

    # Get global model settings
    GLOBAL_MODEL_ADDITIONAL_FEATURES: List[str] = [
        col.lower() for col in ["year", "country name"]
    ]
    GLOBAL_MODEL_TARGETS: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]

    hyperparameters = get_core_hyperparameters(
        input_size=len(LOCAL_MODEL_FEATURES),
        batch_size=32,
        epochs=50,
    )

    # Load whole dataset
    loader = StatesDataLoader()

    # Local model training data
    STATE: str = "Czechia"
    single_state_data_dict = loader.load_states(states=[STATE])
    group_state_data_dict = loader.load_states(states=RICH_STATES)
    all_states_data_dict = loader.load_all_states()

    # Global model taining data
    merged_all_states_data = loader.merge_states(state_dfs=all_states_data_dict)

    PIPELINE_NAME: str = "core_pipeline"
    pipeline = train_basic_pipeline(
        name=PIPELINE_NAME,
        local_model_data_dict=all_states_data_dict,
        global_model_data=merged_all_states_data,
        hyperparameters=hyperparameters,
        local_model_features=LOCAL_MODEL_FEATURES,
        additional_global_model_features=GLOBAL_MODEL_ADDITIONAL_FEATURES,
        global_model_targets=GLOBAL_MODEL_TARGETS,
    )

    # Save pipeline
    pipeline.save_pipeline()

    # Try to predict something, for example Czechia from loaded pipeline
    pipeline = PredictorPipeline.get_pipeline(name=PIPELINE_NAME)

    states_loader = StatesDataLoader()
    czechia_data_dict = states_loader.load_states(states=["Czechia"])

    test_predicion_df = pipeline.predict(
        input_data=czechia_data_dict["Czechia"], target_year=2035
    )

    print(test_predicion_df)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run main
    main()
