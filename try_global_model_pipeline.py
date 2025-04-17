# Standard library imports
from typing import List


# Custom imports
from src.utils.log import setup_logging
from src.pipeline import GlobalModelPipeline
from src.train_scripts.train_global_models import train_global_model
from src.global_model.model import XGBoostTuneParams

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


def main():

    # Targets - yet cannot be just only one of the feature
    # targets: List[str] = [
    #     "population, total",
    #     "population, male",
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
            "Population, total",
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
            # Population total target
        ]
    ]

    loader = StatesDataLoader()
    all_states_data = loader.load_all_states()

    tune_parameters = XGBoostTuneParams(
        n_estimators=[50, 100],
        learning_rate=[0.001, 0.01, 0.05],
        max_depth=[3, 5, 7],
        subsample=[0.5, 0.7],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    pipeline: GlobalModelPipeline = train_global_model(
        states_data=all_states_data,
        features=FEATURES,
        targets=targets,
        sequence_len=5,
        tune_parameters=tune_parameters,
    )

    print("After training pipeline...")
    predictions = pipeline.predict(
        input_data=all_states_data["Czechia"].iloc[:-5],
        last_year=2016,
        target_year=2021,
    )
    print(predictions)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    main()
