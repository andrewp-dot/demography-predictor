# Standard library imports
import pandas as pd
from typing import List, Tuple, Dict

# Custom library imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.base import RNNHyperparameters
from src.evaluation import EvaluateModel

from src.target_model.model import XGBoostTuneParams


from src.state_groups import StatesByWealth

from src.pipeline import PredictorPipeline

from train_scripts.train_feature_models import train_base_rnn
from train_scripts.train_target_models import train_global_model_tree

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


def train_basic_pipeline(
    name: str,
    global_model_data_dict: Dict[str, pd.DataFrame],
    local_model_data_dict: Dict[str, pd.DataFrame],
    hyperparameters: RNNHyperparameters,
    local_model_features: List[str],
    global_model_targets: List[str],
    additional_global_model_features: List[str] = None,
    split_rate: float = 0.8,
    display_nth_epoch=10,
) -> PredictorPipeline:

    # Train local model pipeline
    local_model_pipeline = train_base_rnn(
        name="base-lstm",
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

    global_model_pipeline = train_global_model_tree(
        name="xgb-gm",
        states_data=global_model_data_dict,
        features=list(set([*local_model_features, *additional_global_model_features])),
        targets=global_model_targets,
        sequence_len=5,
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

    # # Get local model features
    # LOCAL_MODEL_FEATURES: List[str] = [
    #     col.lower()
    #     for col in [
    #         # "year",
    #         "Fertility rate, total",
    #         # "Population, total",
    #         "Net migration",
    #         "Arable land",
    #         "Birth rate, crude",
    #         "GDP growth",
    #         "Death rate, crude",
    #         "Agricultural land",
    #         "Rural population",
    #         "Rural population growth",
    #         "Age dependency ratio",
    #         "Urban population",
    #         "Population growth",
    #         "Adolescent fertility rate",
    #         "Life expectancy at birth, total",
    #     ]
    # ]

    LOCAL_MODEL_FEATURES: List[str] = [
        "fertility rate, total",
        "population, total",
        "arable land",
        "gdp growth",
        "death rate, crude",
        "agricultural land",
        "rural population growth",
        "urban population",
        "population growth",
    ]

    # Get global model settings
    GLOBAL_MODEL_ADDITIONAL_FEATURES: List[str] = [
        col.lower() for col in ["year", "country_name"]
    ]
    GLOBAL_MODEL_TARGETS: List[str] = [
        # "population ages 15-64",
        # "population ages 0-14",
        # "population ages 65 and above",
        "population, female",
        "population, male",
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

    # Get the corresponding group of states
    GROUPS_BY_WEALTH = StatesByWealth()

    SELECTED_GROUP: List[str] = GROUPS_BY_WEALTH.get_states_corresponding_group(
        state=STATE
    )

    group_state_data_dict = loader.load_states(states=SELECTED_GROUP)
    all_states_data_dict = loader.load_all_states()

    # Global model taining data
    merged_all_states_data = loader.merge_states(state_dfs=all_states_data_dict)

    PIPELINE_NAME: str = "gender_core_pipeline"
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
    # pipeline.save_pipeline()

    # Try to predict something, for example Czechia from loaded pipeline
    # pipeline = PredictorPipeline.get_pipeline(name=PIPELINE_NAME)

    states_loader = StatesDataLoader()
    czechia_data_dict = states_loader.load_states(states=["Czechia"])

    test_predicion_df = pipeline.predict(
        input_data=czechia_data_dict["Czechia"], target_year=2035
    )
    print(test_predicion_df)

    model_evaluation = EvaluateModel(
        transformer=pipeline.local_model_pipeline.transformer, model=pipeline
    )

    czechia_data_dict = loader.load_states(states=["Czechia"])
    X_test_states, y_test_states = loader.split_data(
        states_dict=czechia_data_dict, sequence_len=hyperparameters.sequence_length
    )

    test_X = X_test_states[STATE]
    test_y = y_test_states[STATE]

    evaluation_df = model_evaluation.eval(test_X=test_X, test_y=test_y)

    print(evaluation_df)

    per_target_df = model_evaluation.eval_per_target(test_X=test_X, test_y=test_y)

    print(per_target_df)

    print(model_evaluation.predicted)

    print(model_evaluation.reference_values)

    pred_plot = model_evaluation.plot_predictions()
    import matplotlib.pyplot as plt

    plt.savefig(
        "predictions.png",
        bbox_inches="tight",
        dpi=300,
    )

    all_states_dict = states_loader.load_all_states()
    # all_states_dict = loader.load_states(states=["Czechia", "United States"])
    X_test_states, y_test_states = loader.split_data(
        states_dict=all_states_dict, sequence_len=hyperparameters.sequence_length
    )

    overall_metrics = model_evaluation.eval_for_every_state_overall(
        X_test_states=X_test_states, y_test_states=y_test_states
    )

    print(overall_metrics)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run main
    main()
