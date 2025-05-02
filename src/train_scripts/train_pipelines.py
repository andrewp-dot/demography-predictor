# Standard library imports
import pandas as pd
from typing import List, Dict
from xgboost import XGBRegressor

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
from src.base import RNNHyperparameters
from src.evaluation import EvaluateModel

from src.target_model.model import XGBoostTuneParams


from src.state_groups import StatesByWealth


from src.pipeline import PredictorPipeline

from src.train_scripts.train_feature_models import (
    train_base_rnn,
    train_arima_ensemble_all_states,
)
from src.train_scripts.train_target_models import train_global_model_tree

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


def train_basic_pipeline(
    name: str,
    target_model_data_dict: Dict[str, pd.DataFrame],
    feature_model_data_dict: Dict[str, pd.DataFrame],
    hyperparameters: RNNHyperparameters,
    feature_model_features: List[str],
    target_model_targets: List[str],
    additional_target_model_targets: List[str] = None,
    split_rate: float = 0.8,
    display_nth_epoch=10,
    enable_early_stopping: bool = True,
) -> PredictorPipeline:

    # Train local model pipeline
    feature_model_pipeline = train_base_rnn(
        name="base-lstm",
        hyperparameters=hyperparameters,
        data=feature_model_data_dict,
        features=feature_model_features,
        split_rate=split_rate,
        display_nth_epoch=display_nth_epoch,
        enable_early_stopping=enable_early_stopping,
    )

    # Train global model pieplien
    tune_parameters = XGBoostTuneParams(
        n_estimators=[50, 100],
        learning_rate=[0.001, 0.01, 0.05],
        max_depth=[3, 5, 7],
        subsample=[0.5, 0.7],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    target_model_pipeline = train_global_model_tree(
        name="xgb-gm",
        states_data=target_model_data_dict,
        tree_model=XGBRegressor(
            n_estimators=100, objective="reg:squarederror", random_state=42
        ),
        features=list(set([*feature_model_features, *additional_target_model_targets])),
        targets=target_model_targets,
        sequence_len=5,
        xgb_tune_parameters=tune_parameters,
    )

    # Create pipeline
    model_pipeline = PredictorPipeline(
        name=name,
        local_model_pipeline=feature_model_pipeline,
        global_model_pipeline=target_model_pipeline,
    )

    return model_pipeline


def train_arima_xgboost_pipeline(
    name: str,
    target_model_data_dict: Dict[str, pd.DataFrame],
    feature_model_data_dict: Dict[str, pd.DataFrame],
    feature_model_features: List[str],
    target_model_targets: List[str],
    additional_target_model_targets: List[str] = None,
    split_rate: float = 0.8,
    display_nth_epoch=10,
) -> PredictorPipeline:
    # Train local model pipeline
    feature_model_pipeline = train_arima_ensemble_all_states(
        name="arima",
        data=feature_model_data_dict,
        features=feature_model_features,
        split_rate=split_rate,
    )

    # Train global model pieplien
    tune_parameters = XGBoostTuneParams(
        n_estimators=[100, 400],
        learning_rate=[0.001, 0.01, 0.05],
        max_depth=[3, 5, 7],
        subsample=[0.5, 0.7],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    target_model_pipeline = train_global_model_tree(
        name="xgb-gm",
        states_data=target_model_data_dict,
        tree_model=XGBRegressor(
            n_estimators=100, objective="reg:squarederror", random_state=42
        ),
        features=list(set([*feature_model_features, *additional_target_model_targets])),
        targets=target_model_targets,
        sequence_len=5,
        xgb_tune_parameters=tune_parameters,
    )

    # Create pipeline
    model_pipeline = PredictorPipeline(
        name=name,
        local_model_pipeline=feature_model_pipeline,
        global_model_pipeline=target_model_pipeline,
    )

    return model_pipeline


def main():

    # Local model features
    FEATURE_MODEL_FEATURES: List[str] = basic_features(
        exclude=highly_correlated_features()
    )

    # Get global model settings
    TARGET_MODEL_ADDITIONAL_FEATURES: List[str] = [col.lower() for col in ["year"]]
    TARGET_MODEL_TARGETS: List[str] = aging_targets()

    hyperparameters = get_core_hyperparameters(
        input_size=len(FEATURE_MODEL_FEATURES),
        # batch_size=1,
        batch_size=32,
        epochs=50,
        sequence_length=5,
    )

    # Load whole dataset
    loader = StatesDataLoader()

    # Local model training data
    STATE: str = "Czechia"
    single_state_data_dict = loader.load_states(states=[STATE])

    # # Get the corresponding group of states
    # GROUPS_BY_WEALTH = StatesByWealth()

    # SELECTED_GROUP: List[str] = GROUPS_BY_WEALTH.get_states_corresponding_group(
    #     state=STATE
    # )
    GROUP_STATES = (
        StatesByWealth().high_income
        + StatesByWealth().upper_middle_income
        + StatesByWealth().lower_middle_income
    )
    group_state_data_dict = loader.load_states(states=GROUP_STATES)
    all_states_data_dict = loader.load_all_states()

    # all_states_data_dict = single_state_data_dict

    # Global model taining data
    # merged_all_states_data = loader.merge_states(state_dfs=all_states_data_dict)

    PIPELINE_NAME: str = "aging_core_pipeline"
    pipeline = train_arima_xgboost_pipeline(
        name=PIPELINE_NAME,
        feature_model_data_dict=single_state_data_dict,
        target_model_data_dict=all_states_data_dict,
        feature_model_features=FEATURE_MODEL_FEATURES,
        additional_target_model_targets=TARGET_MODEL_ADDITIONAL_FEATURES,
        target_model_targets=TARGET_MODEL_TARGETS,
    )

    pipeline_2 = train_basic_pipeline(
        name=PIPELINE_NAME,
        feature_model_data_dict=group_state_data_dict,
        target_model_data_dict=all_states_data_dict,
        hyperparameters=hyperparameters,
        feature_model_features=FEATURE_MODEL_FEATURES,
        additional_target_model_targets=TARGET_MODEL_ADDITIONAL_FEATURES,
        target_model_targets=TARGET_MODEL_TARGETS,
        enable_early_stopping=True,
    )

    # Save pipeline
    pipeline.save_pipeline()

    # Try to predict something, for example Czechia from loaded pipeline
    pipeline = PredictorPipeline.get_pipeline(name=PIPELINE_NAME)

    states_loader = StatesDataLoader()
    czechia_data_dict = states_loader.load_states(states=["Czechia"])


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run main
    main()
