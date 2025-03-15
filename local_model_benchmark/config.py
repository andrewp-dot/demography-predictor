import os
from pydantic import DirectoryPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict


from src.local_model.base import LSTMHyperparameters

from local_model_benchmark.experiments.base_experiment import BaseExperiment


class LocalModelBenchmarkSettings(BaseSettings):

    # Path configs
    benchmark_results_dir: DirectoryPath = Field(
        ...,
        default_factory=lambda: os.path.join(
            ".", "local_model_benchmark", "experiment_results"
        ),
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
