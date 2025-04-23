# Standard library imports
from typing import List


# Custom imports
from config import Config


from src.base import RNNHyperparameters
from src.utils.constants import (
    get_core_hyperparameters,
    basic_features,
    hihgly_correlated_features,
)
from src.state_groups import StatesByWealth
from src.utils.log import setup_logging
from src.pipeline import LocalModelPipeline
from src.train_scripts.train_local_models import train_base_lstm


from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.evaluation import EvaluateModel


settings = Config()

# TARGETS: List[str] = [
#     "population, total",
# ]
TARGETS: List[str] = [
    # "population ages 15-64",
    # "population ages 0-14",
    # "population ages 65 and above",
]

# targets: List[str] = [
#     "population, female",
#     "population, male",
# ]

FEATURES = basic_features(exclude=hihgly_correlated_features)


def train_only_features(epochs: int):

    loader = StatesDataLoader()
    # states_data_dict = loader.load_all_states()
    states_data_dict = loader.load_states(
        states=[
            *StatesByWealth().high_income,
            *StatesByWealth().upper_middle_income,
            *StatesByWealth().lower_middle_income,
        ]
    )

    # custom_features = FEATURES
    custom_features = FEATURES

    hyperparameters: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(custom_features),
        batch_size=32,
        epochs=epochs,
        hidden_size=128,
        future_step_predict=1,
    )

    pipeline = train_base_lstm(
        name="lstm_features_only",
        hyperparameters=hyperparameters,
        data=states_data_dict,
        features=custom_features,
    )

    pipeline.save_pipeline()


def train_including_targets(epochs: int):

    loader = StatesDataLoader()
    # states_data_dict = loader.load_all_states()
    states_data_dict = loader.load_states(
        states=[
            *StatesByWealth().high_income,
            *StatesByWealth().upper_middle_income,
            *StatesByWealth().lower_middle_income,
        ]
    )

    # custom_features = FEATURES
    custom_features = TARGETS
    # custom_features = FEATURES + TARGETS

    hyperparameters: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(custom_features),
        batch_size=32,
        epochs=epochs,
        hidden_size=128,
        future_step_predict=2,
    )

    pipeline = train_base_lstm(
        name="lstm_features_targets",
        hyperparameters=hyperparameters,
        data=states_data_dict,
        features=custom_features,
    )

    pipeline.save_pipeline()


def evaluate_pipelines():

    # Load data
    loader = StatesDataLoader()
    states_data_dict = loader.load_all_states()

    states_data_dict = loader.load_states(
        states=["Czechia", "United States", "Honduras"]
    )

    # Load pipelines
    features_only_pipeline = LocalModelPipeline.get_pipeline(name="lstm_features_only")
    including_targets_pipeline = LocalModelPipeline.get_pipeline(
        name="lstm_features_targets"
    )

    # Split data
    f_only_X_test_dict, f_only_y_test_dict = loader.split_data(
        states_dict=states_data_dict,
        sequence_len=features_only_pipeline.model.hyperparameters.sequence_length,
        future_steps=features_only_pipeline.model.hyperparameters.future_step_predict,
    )

    tragets_included_X_test_dict, tragets_included_y_test_dict = loader.split_data(
        states_dict=states_data_dict,
        sequence_len=features_only_pipeline.model.hyperparameters.sequence_length,
        future_steps=features_only_pipeline.model.hyperparameters.future_step_predict,
    )

    features_only_eval = EvaluateModel(pipeline=features_only_pipeline)
    including_targets_eval = EvaluateModel(pipeline=including_targets_pipeline)

    f_only_df = features_only_eval.eval_for_every_state(
        X_test_states=f_only_X_test_dict, y_test_states=f_only_y_test_dict
    )

    eval_targets_df = including_targets_eval.eval_for_every_state(
        X_test_states=tragets_included_X_test_dict,
        y_test_states=tragets_included_y_test_dict,
    )

    # Sort them by
    f_only_df.sort_values(by=["r2", "mae"], inplace=True)
    eval_targets_df.sort_values(by=["r2", "mae"], ascending=[True, False])

    with open("features_only.json", "w") as f:
        f_only_df.to_json(f, indent=4, orient="records")

    with open("features_targets_included.json", "w") as f:
        eval_targets_df.to_json(f, indent=4, orient="records")

    print(f_only_df)
    print("-" * 100)
    print(eval_targets_df)


if __name__ == "__main__":

    # Setup logging
    setup_logging()

    EPOCHS: int = 10

    # future_step_predict=1,
    train_only_features(epochs=EPOCHS)

    # future_step_predict=2,
    # train_including_targets(epochs=EPOCHS)

    evaluate_pipelines()
