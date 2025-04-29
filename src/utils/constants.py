# Custom library imports
from src.base import RNNHyperparameters
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
) -> RNNHyperparameters:
    BASE_HYPERPARAMETERS: RNNHyperparameters = RNNHyperparameters(
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
def highly_correlated_features() -> List[str]:
    return [
        "life_expectancy_at_birth_total",
        "age_dependency_ratio",
        "rural_population",
        "birth_rate_crude",
        "adolescent_fertility_rate",
    ]


def basic_features(exclude: List[str] | None = None) -> List[str]:
    return [
        col.lower()
        for col in [
            "year",
            "fertility_rate_total",
            "net_migration",
            "arable_land",
            "birth_rate_crude",
            "gdp_growth",
            "gdp",
            "death_rate_crude",
            "agricultural_land",
            "rural_population",
            "rural_population_growth",
            "age_dependency_ratio",
            "urban_population",
            "population_growth",
            "adolescent_fertility_rate",
            "life_expectancy_at_birth_total",
        ]
        if col not in exclude
    ]


# Targets constants
def aging_targets() -> List[str]:
    return [
        "population_ages_15-64",
        "population_ages_0-14",
        "population_ages_65_and_above",
    ]


def gender_distribution_targets() -> List[str]:
    return [
        "population_female",
        "population_male",
    ]


def population_total_targets() -> List[str]:
    return ["population_total"]


def categorical_columns() -> List[str]:
    return ["country_name"]


def absolute_columns() -> List[str]:
    return [
        # Features
        "year",
        "fertility_rate_total",
        "birth_rate_crude",
        "adolescent_fertility_rate",
        "death_rate_crude",
        "life_expectancy_at_birth_total",
    ]


def percentual_columns() -> List[str]:
    return [
        # Features
        "population_growth",
        "arable_land",
        "gdp_growth",
        "agricultural_land",
        "rural_population",
        "rural_population_growth",
        "urban_population",
        "age_dependency_ratio",
        # Targets
        "population_ages_15-64",
        "population_ages_0-14",
        "population_ages_65_and_above",
        "population_female",
        "population_male",
    ]


def wide_range_columns() -> List[str]:
    return [
        "net_migration",
        "population_total",
        "gdp",
    ]
