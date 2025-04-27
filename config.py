import os
from pydantic import Field, DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Dict, List


# Custom settings
DATASET_VERSION: Literal["v0", "v1", "v2", "v3"] = "v3"

ALL_POSSIBLE_TARGET_FEATURES: List[str] = [
    "population_total",
    "population_ages_15-64",
    "population_ages_0-14",
    "population_ages_65_and_above",
    "population_female",
    "population_male",
]


def get_trained_models_dir() -> DirectoryPath:
    trained_models_dir = os.path.join(os.path.abspath("."), "trained_models")

    if not os.path.isdir(trained_models_dir):
        os.makedirs(trained_models_dir)

    return trained_models_dir


class Config(BaseSettings):

    # Configure settings config
    model_config: dict = SettingsConfigDict(env_file=".env", frozen=True)

    # Dataset settings
    selected_dataset: str = f"dataset_{DATASET_VERSION}"
    dataset_dir: str = os.path.join(".", "data_science", "datasets", selected_dataset)
    dataset_path: str = os.path.join(dataset_dir, f"{selected_dataset}.csv")
    states_data_dir: str = os.path.join(dataset_dir, "states")

    # Local model benchmark settings
    benchmark_results_dir: str = os.path.join(
        ".", "local_model_benchmark", "experiment_results"
    )

    # Model save dir
    trained_models_dir: str = Field(..., default_factory=get_trained_models_dir)

    # API configs
    api_host: str = Field(..., alias="API_HOST")
    api_port: int = Field(..., alias="API_PORT")

    # Load model configs
    @computed_field
    def prediction_models(self) -> Dict[str, str]:
        return {
            # model_name: f"pipeline_name"
            "aging_model": "core_pipeline",
            "aging_model_rich": "group_pipeline",
            "aging_model_cz": "cz_pipeline",
            "gender_model": "gender_core_pipeline",
        }

    ALL_POSSIBLE_TARGET_FEATURES: List[str] = ALL_POSSIBLE_TARGET_FEATURES


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


def get_data_visualizations_dir(
    data_science_root: DirectoryPath, dataset_version: str
) -> DirectoryPath:
    data_visualizations_dir = os.path.join(
        data_science_root,
        "visualizations",
        f"dataset_{dataset_version}",
    )

    # Create if it does not exist
    if not os.path.isdir(data_visualizations_dir):
        os.makedirs(data_visualizations_dir)

    return data_visualizations_dir


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

    visualizations_dir: DirectoryPath = Field(
        ...,
        default_factory=lambda: get_data_visualizations_dir(
            get_data_science_dir(), DATASET_VERSION
        ),
    )

    ALL_POSSIBLE_TARGET_FEATURES: List[str] = ALL_POSSIBLE_TARGET_FEATURES
