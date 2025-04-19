# Standard library imports
from typing import List


# Custom imports
from config import Config

from src.state_groups import StatesByWealth
from src.utils.log import setup_logging
from src.pipeline import GlobalModelPipeline
from src.train_scripts.train_global_models import train_global_model
from src.global_model.model import XGBoostTuneParams

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.evaluation import EvaluateModel


settings = Config()


def train_pipeline(name: str, sequence_len: int):

    # Targets - yet cannot be just only one of the feature
    # targets: List[str] = [
    #     "population, total",
    # ]
    targets: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]

    # targets: List[str] = [
    #     "population, female",
    #     "population, male",
    # ]

    # Features
    # Features
    FEATURES = [
        col.lower()
        for col in [
            # "year",
            "Fertility rate, total",
            # "population, total",
            "Net migration",
            "Arable land",
            # "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            # "Rural population",
            "Rural population growth",
            # "Age dependency ratio",
            "Urban population",
            "Population growth",
            # "Adolescent fertility rate",
            # "Life expectancy at birth, total",
        ]
    ]

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
    NAME = "test_gm"

    train_pipeline(name=NAME, sequence_len=SEQUENCE_LEN)

    eval_pipeline(name=NAME, sequence_len=SEQUENCE_LEN)
