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
    all_states_dict = loader.load_all_states()

    hyperparameters = get_core_hyperparameters(input_size=len(FEATURES), epochs=10)

    predictor_pipeline = train_basic_pipeline(
        name="test_predictor",
        global_model_data_dict=all_states_dict,
        global_model_targets=TARGETS,
        local_model_data_dict=all_states_dict,
        local_model_features=FEATURES,
        hyperparameters=hyperparameters,
        additional_global_model_features=["year", "country name"],
    )

    predictor_pipeline.save_pipeline()


def eval():

    loader = StatesDataLoader()
    test_data_dict = loader.load_states(states=["Czechia", "Honduras", "United States"])

    pipeline = PredictorPipeline.get_pipeline("test_predictor")

    X_test_states, y_test_states = loader.split_data(
        test_data_dict,
        sequence_len=pipeline.local_model_pipeline.model.hyperparameters.sequence_length,
    )

    evaluation = EvaluateModel(pipeline=pipeline)

    eval_df = evaluation.eval_for_every_state_overall(
        X_test_states=X_test_states, y_test_states=y_test_states
    )
    print(eval_df)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    main()

    eval()
