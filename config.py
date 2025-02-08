from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    # configure settings config
    model_config: dict = SettingsConfigDict(frozen=True)

    # dataset settings
    selected_dataset: str = "dataset_v1"
    dataset_dir: str = f"./data_science/datasets/{selected_dataset}"
    dataset_path: str = f"{dataset_dir}/{selected_dataset}.csv"
    states_data_dir: str = f"{dataset_dir}/states"
