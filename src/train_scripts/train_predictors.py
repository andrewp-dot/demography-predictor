# Standard library imports
import pandas as pd
from typing import List, Dict, Literal, Optional
import logging


# Custom library imports
from src.utils.log import setup_logging
from src.utils.constants import (
    basic_features,
    highly_correlated_features,
    aging_targets,
    gender_distribution_targets,
    population_total_targets,
)
from src.utils.constants import get_core_hyperparameters

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.state_groups import StatesByWealth, StatesByGeolocation, OnlyRealStates


from src.pipeline import PredictorPipeline

from src.train_scripts.train_pipelines import (
    train_arima_xgboost_pipeline,
    train_basic_pipeline,
)

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger("training")


def get_training_data(
    wealth_groups: Optional[List[str]] = None,
    geolocation_groups: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    model_type: str = "feature_model",
    loader: Optional[StatesDataLoader] = None,
) -> Dict[str, pd.DataFrame]:

    if not loader:
        loader = StatesDataLoader()

    TRAIN_STATES: List[str] = []
    # Process wealth groups
    if wealth_groups:
        WEALTH_GROUPS = StatesByWealth()
        wealth_groups_list = list(StatesByWealth.model_fields.keys())

        wealth_states = []
        for group in set(wealth_groups):

            # Check if it is there
            if not hasattr(WEALTH_GROUPS, group):
                raise ValueError(
                    f"Group '{group}' is not recognized. Available groups: {wealth_groups_list}"
                )

            wealth_states += getattr(WEALTH_GROUPS, group)

        TRAIN_STATES += wealth_states

    if geolocation_groups:
        GEOLOCATION_GROUPS = StatesByGeolocation()
        geolocation_groups_list = list(StatesByWealth.model_fields.keys())

        geolocation_group_states = []
        for group in set(geolocation_groups):

            # Check if it is there
            if not hasattr(GEOLOCATION_GROUPS, group):
                raise ValueError(
                    f"Group '{group}' is not recognized. Available groups: {geolocation_groups_list}"
                )

            geolocation_group_states += getattr(GEOLOCATION_GROUPS, group)

        TRAIN_STATES += wealth_states

    if states:
        TRAIN_STATES += states

    # Remove duplicates
    TRAIN_STATES = list(set(TRAIN_STATES))

    # Load data based on states
    if TRAIN_STATES:
        logger.info(f"Using states for '{model_type}': {TRAIN_STATES}")
        return loader.load_states(states=TRAIN_STATES)

    logger.info(f"Using all available data for trainig")
    return loader.load_all_states()


def train_aging_predictor(
    name: str,
    model_type: Literal["LSTM", "ARIMA"],
    wealth_groups: Optional[List[str]] = None,
    geolocation_groups: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    modify_for_target_model: bool = False,
):
    # Feature model input/output
    FEATURE_MODEL_FEATURES: List[str] = basic_features(
        exclude=highly_correlated_features()
    )

    # Get global model settings
    TARGET_MODEL_ADDITIONAL_FEATURES: List[str] = [col.lower() for col in ["year"]]
    TARGET_MODEL_TARGETS: List[str] = aging_targets()

    hyperparameters = get_core_hyperparameters(
        input_size=len(FEATURE_MODEL_FEATURES),
        batch_size=32,
        epochs=50,
        sequence_length=10,
        num_layers=2,
        hidden_size=256,
    )

    loader = StatesDataLoader()

    # Get modified training data
    training_states_data_dict = get_training_data(
        wealth_groups=wealth_groups,
        geolocation_groups=geolocation_groups,
        states=states,
    )

    # Get all training data
    all_states_data_dict = loader.load_all_states()

    # Make the modification for one of the models
    if modify_for_target_model:
        feature_model_data_dict = all_states_data_dict
        target_model_data_dict = training_states_data_dict
    else:
        feature_model_data_dict = training_states_data_dict
        target_model_data_dict = all_states_data_dict

    # Train pipeline based on type
    PIPELINE_NAME: str = name

    if "ARIMA" == model_type:
        pipeline = train_arima_xgboost_pipeline(
            name=PIPELINE_NAME,
            feature_model_data_dict=feature_model_data_dict,
            target_model_data_dict=target_model_data_dict,
            feature_model_features=FEATURE_MODEL_FEATURES,
            additional_target_model_targets=TARGET_MODEL_ADDITIONAL_FEATURES,
            target_model_targets=TARGET_MODEL_TARGETS,
            tune_hyperparams=True,
        )

    elif "LSTM" == model_type:
        pipeline = train_basic_pipeline(
            name=PIPELINE_NAME,
            feature_model_data_dict=training_states_data_dict,
            target_model_data_dict=training_states_data_dict,
            hyperparameters=hyperparameters,
            feature_model_features=FEATURE_MODEL_FEATURES,
            additional_target_model_targets=TARGET_MODEL_ADDITIONAL_FEATURES,
            target_model_targets=TARGET_MODEL_TARGETS,
            enable_early_stopping=True,
        )
    else:
        raise ValueError("Not supported type of pipeline.")

    # Save pipeline
    pipeline.save_pipeline()


def train_gender_dist_predictor(
    name: str,
    model_type: Literal["LSTM", "ARIMA"],
    wealth_groups: Optional[List[str]] = None,
    geolocation_groups: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    modify_for_target_model: bool = False,
):
    # Feature model input/output
    FEATURE_MODEL_FEATURES: List[str] = basic_features(
        exclude=highly_correlated_features()
    )

    # Get global model settings
    TARGET_MODEL_ADDITIONAL_FEATURES: List[str] = [col.lower() for col in ["year"]]
    TARGET_MODEL_TARGETS: List[str] = gender_distribution_targets()

    hyperparameters = get_core_hyperparameters(
        input_size=len(FEATURE_MODEL_FEATURES),
        batch_size=32,
        epochs=50,
        sequence_length=10,
        num_layers=2,
        hidden_size=256,
    )

    loader = StatesDataLoader()

    # Get modified training data
    training_states_data_dict = get_training_data(
        wealth_groups=wealth_groups,
        geolocation_groups=geolocation_groups,
        states=states,
    )

    # Get all training data
    all_states_data_dict = loader.load_all_states()

    # Make the modification for one of the models
    if modify_for_target_model:
        feature_model_data_dict = all_states_data_dict
        target_model_data_dict = training_states_data_dict
    else:
        feature_model_data_dict = training_states_data_dict
        target_model_data_dict = all_states_data_dict

    # Train pipeline based on type
    PIPELINE_NAME: str = name

    if "ARIMA" == model_type:
        pipeline = train_arima_xgboost_pipeline(
            name=PIPELINE_NAME,
            feature_model_data_dict=feature_model_data_dict,
            target_model_data_dict=target_model_data_dict,
            feature_model_features=FEATURE_MODEL_FEATURES,
            additional_target_model_targets=TARGET_MODEL_ADDITIONAL_FEATURES,
            target_model_targets=TARGET_MODEL_TARGETS,
            tune_hyperparams=True,
        )

    elif "LSTM" == model_type:
        pipeline = train_basic_pipeline(
            name=PIPELINE_NAME,
            feature_model_data_dict=training_states_data_dict,
            target_model_data_dict=training_states_data_dict,
            hyperparameters=hyperparameters,
            feature_model_features=FEATURE_MODEL_FEATURES,
            additional_target_model_targets=TARGET_MODEL_ADDITIONAL_FEATURES,
            target_model_targets=TARGET_MODEL_TARGETS,
            enable_early_stopping=True,
        )
    else:
        raise ValueError("Not supported type of pipeline.")

    # Save pipeline
    pipeline.save_pipeline()
