# Custom library imports
from src.base import LSTMHyperparameters
from typing import List


# Create or load base model
def get_core_hyperparameters(
    input_size: int,
    hidden_size: int = 512,
    future_step_predict: int = 1,
    sequence_length: int = 10,
    learning_rate: float = 0.0001,
    epochs: int = 10,
    batch_size: int = 1,
    num_layers: int = 3,
    output_size: int | None = None,
    bidirectional: bool = False,
) -> LSTMHyperparameters:
    BASE_HYPERPARAMETERS: LSTMHyperparameters = LSTMHyperparameters(
        input_size=input_size,
        hidden_size=hidden_size,
        future_step_predict=future_step_predict,
        output_size=output_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    return BASE_HYPERPARAMETERS


# Feature constants
def hihgly_correlated_features() -> List[str]:
    return [
        "life expectancy at birth, total",
        "age dependency ratio",
        "rural population",
        "birth rate, crude",
        "adolescent fertility rate",
    ]


def basic_features(exclude: List[str] | None = None) -> List[str]:
    return [
        "fertility rate, total",
        # "population, total",
        "arable land",
        "gdp growth",
        "death rate, crude",
        "agricultural land",
        "rural population growth",
        "urban population",
        "population growth",
        # col.lower()
        # for col in [
        #     "year",
        #     "Fertility rate, total",
        #     "population, total",
        #     "Net migration",
        #     "Arable land",
        #     "Birth rate, crude",
        #     "GDP growth",
        #     "Death rate, crude",
        #     "Agricultural land",
        #     "Rural population",
        #     "Rural population growth",
        #     "Age dependency ratio",
        #     "Urban population",
        #     "Population growth",
        #     "Adolescent fertility rate",
        #     "Life expectancy at birth, total",
        # ]
        # if col not in exclude
    ]


# Targets constants
def aging_targets() -> List[str]:
    return [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]


def gender_distribution_targets() -> List[str]:
    return [
        "population, female",
        "population, male",
    ]


def population_total_targets() -> List[str]:
    return ["population, total"]
