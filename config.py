import os
from pydantic import Field, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


def get_trained_models_dir() -> DirectoryPath:
    trained_models_dir = os.path.join(".", "trained_models")

    if not os.path.isdir(trained_models_dir):
        os.makedirs(trained_models_dir)

    return trained_models_dir


class Config(BaseSettings):

    # Configure settings config
    model_config: dict = SettingsConfigDict(frozen=True)

    # Dataset settings
    selected_dataset: str = "dataset_v1"
    dataset_dir: str = os.path.join(".", "data_science", "datasets", selected_dataset)
    dataset_path: str = os.path.join(dataset_dir, f"{selected_dataset}.csv")
    states_data_dir: str = os.path.join(dataset_dir, "states")

    # Local model benchmark settings
    benchmark_results_dir: str = os.path.join(
        ".", "local_model_benchmark", "experiment_results"
    )

    # Model save dir
    trained_models_dir: str = Field(..., default_factory=get_trained_models_dir)


def get_data_science_dir() -> DirectoryPath:
    return os.path.abspath(os.path.join(".", "data_science"))


def get_dataset_dir(data_science_root: DirectoryPath) -> DirectoryPath:
    return os.path.join(data_science_root, "datasets")


def get_data_dir(data_science_root: DirectoryPath) -> DirectoryPath:
    return os.path.join(data_science_root, "data")


def get_source_data_dir(
    data_science_root: DirectoryPath, dataset_version: str
) -> DirectoryPath:
    return os.path.join(data_science_root, "data", f"dataset_{dataset_version}")


# Custom settings
DATASET_VERSION: Literal["v0", "v1"] = "v1"


class DatasetCreatorSettings(BaseSettings):

    dataset_version: str = DATASET_VERSION
    model_config = SettingsConfigDict(frozen=True)

    # Path configs
    source_data_dir: DirectoryPath = Field(
        ...,
        default_factory=lambda: get_source_data_dir(
            get_data_science_dir(), DATASET_VERSION
        ),
    )

    save_dataset_path: DirectoryPath = Field(
        ..., default_factory=lambda: get_dataset_dir(get_data_science_dir())
    )

    data_dir: DirectoryPath = Field(
        ..., default_factory=lambda: get_data_dir(get_data_science_dir())
    )
