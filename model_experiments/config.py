import os
from pydantic import DirectoryPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict

from src.base import RNNHyperparameters


def get_experiment_result_dir() -> DirectoryPath:
    experiment_result_dir = os.path.join(".", "model_experiments", "experiment_results")

    # Create it if it does not exist
    if not os.path.isdir(experiment_result_dir):
        os.makedirs(experiment_result_dir)

    return experiment_result_dir


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
