# Standard library imports
import logging
from typing import List


# Custom imports
from config import Config

from src.utils.log import setup_logging

from src.utils.constants import (
    get_core_hyperparameters,
    basic_features,
    hihgly_correlated_features,
    aging_targets,
)

from src.base import TrainingStats

from src.state_groups import StatesByWealth

from src.pipeline import GlobalModelPipeline
from src.train_scripts.train_global_models import train_global_model
from src.global_model.model import XGBoostTuneParams
from src.global_model.global_rnn import GlobalModelRNN

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.evaluation import EvaluateModel


logger = logging.getLogger("global_model")

settings = Config()


def train_pipeline(name: str, sequence_len: int):

    targets: List[str] = aging_targets()

    # Features
    FEATURES: List[str] = basic_features(exclude=hihgly_correlated_features())

    loader = StatesDataLoader()
    # all_states_data = loader.load_all_states()
    all_states_data = loader.load_states(
        states=[
            *StatesByWealth().high_income,
            *StatesByWealth().upper_middle_income,
            *StatesByWealth().lower_middle_income,
            # *StatesByWealth().low_income,
        ]
        # states=[
        #     "Honduras",
        #     "Czechia",
        #     "United States",
        # ]
    )

    tune_parameters = XGBoostTuneParams(
        n_estimators=[200, 400],
        learning_rate=[0.01, 0.05, 0.1],
        max_depth=[3, 5, 7],
        subsample=[0.8, 1.0],
        colsample_bytree=[0.8, 1.0],
    )

    pipeline: GlobalModelPipeline = train_global_model(
        name=name,
        states_data=all_states_data,
        features=FEATURES,
        targets=targets,
        sequence_len=sequence_len,
        tune_parameters=tune_parameters,
    )

    pipeline.save_pipeline()

    print("After training pipeline...")

    print("-" * 100)
    print(all_states_data["Czechia"][[*targets, "year"]])
    print("-" * 100)

    # Here is a problem
    predictions = pipeline.predict(
        input_data=all_states_data["Czechia"],
        last_year=2016,
        target_year=2021,
    )
    print(predictions)

    print(all_states_data["Czechia"].iloc[-5:][[*targets, "year"]])


def train_rnn_pipeline(
    name: str, sequence_len: int = 10, save_plots: bool = True, epochs: int = 50
):
    # Setup logging
    setup_logging()

    FEATURES = basic_features(exclude=hihgly_correlated_features())

    TARGETS: List[str] = aging_targets()

    # WHOLE_MODEL_FEATURES: List[str] = FEATURES + TARGETS

    # Setup model - on input it gets the past sequences of features and also past sequences of the target variable
    hyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES + TARGETS),  # To input include history with targets
        epochs=epochs,
        batch_size=32,
        output_size=len(TARGETS),
        sequence_length=sequence_len,
    )
    rnn = GlobalModelRNN(hyperparameters, features=FEATURES, targets=TARGETS)

    # Load data
    states_loader = StatesDataLoader()
    states_data_dict = states_loader.load_all_states()

    states_data_dict = states_loader.load_states(
        states=[
            *StatesByWealth().high_income,
            *StatesByWealth().upper_middle_income,
            *StatesByWealth().lower_middle_income,
            # *StatesByWealth().low_income,
        ]
    )

    # states_data_dict = states_loader.load_states(states=StatesByWealth().high_income)

    # Get training data and validation data
    train_data_dict, test_data_dict = states_loader.split_data(
        states_dict=states_data_dict, sequence_len=hyperparameters.sequence_length
    )
    train_states_df = states_loader.merge_states(train_data_dict)
    test_states_df = states_loader.merge_states(test_data_dict)

    # Transform data
    transformer = DataTransformer()

    # This is useless maybe -> just for fitting the scaler on training data after transformation in datatransforme
    scaled_training_data, scaled_test_data = transformer.scale_and_fit(
        training_data=train_states_df,
        validation_data=test_states_df,
        features=FEATURES,
        targets=TARGETS,
    )

    # Create a dictionary from it
    scaled_states_dict = states_loader.parse_states(scaled_training_data)

    batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets = (
        transformer.create_train_test_multiple_states_batches(
            data=scaled_states_dict,
            hyperparameters=hyperparameters,
            features=FEATURES + TARGETS,
            targets=TARGETS,
        )
    )

    print("-" * 100)
    logger.info(f"Batch inputs shape: {batch_inputs.shape}")
    logger.info(f"Batch targets shape: {batch_targets.shape}")

    # Train model
    stats = rnn.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=1,
    )

    training_stats = TrainingStats.from_dict(stats_dict=stats)

    if save_plots:
        fig = training_stats.create_plot()
        fig.savefig(f"global_rnn_{hyperparameters.epochs}_epochs.png")

    pipeline = GlobalModelPipeline(name=name, model=rnn, transformer=transformer)
    # save_model(name=f"global_rnn.pkl", model=rnn)
    # save_model(name=f"global_rnn_transformer.pkl", model=transformer)

    pipeline.save_pipeline()


def eval_pipeline(name: str, sequence_len: int):

    # Evaluation
    pipeline = GlobalModelPipeline.get_pipeline(name=name)
    gmeval = EvaluateModel(pipeline=pipeline)

    loader = StatesDataLoader()
    states_data_dict = loader.load_states(
        states=StatesByWealth().high_income
        # states=[
        #     "Honduras",
        #     "Czechia",
        #     "United States",
        # ]
    )

    states_data_dict = loader.load_all_states()
    train_states, test_states = loader.split_data(
        states_dict=states_data_dict, sequence_len=sequence_len
    )

    every_state_eval = gmeval.eval_for_every_state(
        X_test_states=train_states, y_test_states=test_states
    )

    import matplotlib.pyplot as plt

    gmeval.plot_predictions()

    plt.savefig("gm_test.png")

    every_state_eval.sort_values(by=["r2", "mae"], inplace=True)

    with open("global_model_eval.json", "w") as f:
        every_state_eval.to_json(f, indent=4, orient="records")

    print(every_state_eval)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    SEQUENCE_LEN = 10
    # NAME = "test_gm"
    NAME = "global_model_rnn"

    # train_pipeline(name=NAME, sequence_len=SEQUENCE_LEN)

    train_rnn_pipeline(name=NAME, epochs=30)
    eval_pipeline(name=NAME, sequence_len=SEQUENCE_LEN)
