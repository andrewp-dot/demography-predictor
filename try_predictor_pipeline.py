# Standard library imports
from typing import List


# Custom imports
from config import Config

from src.utils.constants import get_core_hyperparameters, basic_features, aging_targets
from src.state_groups import StatesByWealth
from src.utils.log import setup_logging
from src.pipeline import PredictorPipeline
from src.train_scripts.train_pipelines import train_basic_pipeline


from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.evaluation import EvaluateModel


settings = Config()


# TODO: write exclusion target
TARGETS: List[str] = aging_targets()
FEATURES = basic_features(exclude=["year"])


def main():
    loader = StatesDataLoader()
    # all_states_dict = loader.load_all_states()

    all_states_dict = loader.load_states(
        states=[
            *StatesByWealth().high_income,
            *StatesByWealth().upper_middle_income,
            *StatesByWealth().lower_middle_income,
        ]
    )

    hyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES),
        epochs=20,
        hidden_size=256,
        batch_size=32,
    )

    predictor_pipeline = train_basic_pipeline(
        name="test_predictor",
        global_model_data_dict=all_states_dict,
        global_model_targets=TARGETS,
        local_model_data_dict=all_states_dict,
        local_model_features=FEATURES,
        hyperparameters=hyperparameters,
        additional_global_model_features=["year", "country name"],
        display_nth_epoch=2,
    )

    predictor_pipeline.save_pipeline()


def eval():

    loader = StatesDataLoader()
    # test_data_dict = loader.load_states(states=["Czechia", "Honduras", "United States"])
    # test_data_dict = loader.load_all_states()

    test_data_dict = loader.load_states(
        states=[
            *StatesByWealth().high_income,
            *StatesByWealth().upper_middle_income,
            *StatesByWealth().lower_middle_income,
        ]
    )

    pipeline = PredictorPipeline.get_pipeline("test_predictor")

    X_test_states, y_test_states = loader.split_data(
        test_data_dict,
        sequence_len=pipeline.local_model_pipeline.model.hyperparameters.sequence_length,
    )

    evaluation = EvaluateModel(pipeline=pipeline)

    eval_df = evaluation.eval_for_every_state(
        X_test_states=X_test_states, y_test_states=y_test_states
    )

    # Save to json file
    with open("test_pred_eval_all_states_20.json", "w") as f:

        # Sort
        eval_df.sort_values(by=["r2", "mse"], ascending=[False, True], inplace=True)
        eval_df.to_json(f, indent=4, orient="records")

    # Print top 5 states and the worst 5 states

    print(f"Best performance states:\n{eval_df.head(5)}")

    print(f"Worst performance for states:\n{eval_df.tail(5)}")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # main()

    eval()
