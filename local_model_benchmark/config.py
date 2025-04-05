import os
from pydantic import DirectoryPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict

from src.base import LSTMHyperparameters


def get_experiment_result_dir() -> DirectoryPath:
    experiment_result_dir = os.path.join(
        ".", "local_model_benchmark", "experiment_results"
    )

    # Create it if it does not exist
    if not os.path.isdir(experiment_result_dir):
        os.makedirs(experiment_result_dir)

    return experiment_result_dir


def get_core_parameters(
    input_size: int,
    hidden_size: int = 256,
    sequence_length: int = 10,
    learning_rate: float = 0.0001,
    epochs: int = 10,
    batch_size: int = 1,
    num_layers: int = 3,
) -> LSTMHyperparameters:
    BASE_HYPERPARAMETERS: LSTMHyperparameters = LSTMHyperparameters(
        input_size=input_size,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_layers=num_layers,
    )

    return BASE_HYPERPARAMETERS


class LocalModelBenchmarkSettings(BaseSettings):

    # Path configs
    benchmark_results_dir: DirectoryPath = Field(
        ..., default_factory=get_experiment_result_dir
    )

    # Model config
    model_config: Dict = SettingsConfigDict(frozen=True)

    ALL_FEATURES: List = [
        col.lower()
        for col in [
            "year",
            "Fertility rate, total",
            "Population, total",
            "Net migration",
            "Arable land",
            "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            "Rural population",
            "Rural population growth",
            "Age dependency ratio",
            "Urban population",
            "Population growth",
            "Adolescent fertility rate",
            "Life expectancy at birth, total",
        ]
    ]
