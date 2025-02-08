import os
from pydantic import Field, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


# custom settings
DATASET_VERSION: Literal["v0", "v1"] = "v1"


def get_root_dir() -> DirectoryPath:
    return os.path.abspath("./")


def get_dataset_dir(root: DirectoryPath) -> DirectoryPath:
    return os.path.join(root, "../")


def get_data_dir(root: DirectoryPath) -> DirectoryPath:
    return os.path.join(root, "../", "data")


def get_source_data_dir(root: DirectoryPath, dataset_version: str) -> DirectoryPath:
    return os.path.join(root, "../", "data", f"dataset_{dataset_version}")


# TODO: implement necessary settings
# place dataset in here or not?
class Settings(BaseSettings):

    dataset_version: str = DATASET_VERSION
    model_config = SettingsConfigDict(frozen=True)

    # Path configs
    source_data_dir: DirectoryPath = Field(
        ...,
        default_factory=lambda: get_source_data_dir(get_root_dir(), DATASET_VERSION),
    )

    save_dataset_path: DirectoryPath = Field(
        ..., default_factory=lambda: get_dataset_dir(get_root_dir())
    )

    data_dir: DirectoryPath = Field(
        ..., default_factory=lambda: get_data_dir(get_root_dir())
    )
