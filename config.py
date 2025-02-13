import logging
import logging.config
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def setup_logging() -> None:
    # Load YAML config
    try:
        with open("loggers_config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Apply logging configuration
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Failed to setup logging: {e}!")


class Config(BaseSettings):

    # configure settings config
    model_config: dict = SettingsConfigDict(frozen=True)

    # dataset settings
    selected_dataset: str = "dataset_v1"
    dataset_dir: str = f"./data_science/datasets/{selected_dataset}"
    dataset_path: str = f"{dataset_dir}/{selected_dataset}.csv"
    states_data_dir: str = f"{dataset_dir}/states"
