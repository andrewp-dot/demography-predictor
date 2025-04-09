# Standard libraries imports
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import logging

# Custom libraries imports
from config import Config
from src.utils.log import setup_logging

logger = logging.getLogger("data_preprocessing")

config = Config()


# Simplification for data loading
class StateDataLoader:

    def __init__(self, state: str):
        self.state = state

    def load_data(self, exclude_covid_years: bool = False) -> pd.DataFrame:
        """
        Load the state data. Renames the columns all to lower.

        Retruns:
            out: pd.DataFrame: Loaded states data sorted by 'year' in ascending order
        """
        data = pd.read_csv(f"{config.states_data_dir}/{self.state}.csv")

        # Lowercase columns in case they are not
        data.columns = data.columns.str.lower()

        if exclude_covid_years:
            # Covid started about 17th November in 2019. The impact of covid 19 is expected on years 2020 and higher.
            data = data.loc[data["year"] < 2020]

        return data.sort_values(by="year", ascending=True)

    def get_last_year(self, data: pd.DataFrame) -> int:
        """
        Get the last record in the dataframme.

        Args:
            data (pd.DataFrame): Dataframe which should contain the year.

        Returns:
            out: int: Last year of the record.
        """

        # Verify if the 'year' is in the columns
        if "year" not in data.columns:
            raise ValueError("The given dataframe does not contain the 'year' column!")

        # Retrieve the last year from data
        last_year = int(data["year"].sort_values(ascending=False).iloc[0])

        return last_year

    def split_data(
        self, data: pd.DataFrame, split_rate: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train and test sets. Data are expected to be sorted by year in ascending order.

        :param data: pd.DataFrame
        :return: Tuple[pd.DataFrame, pd.DataFrame]: train and test data
        """

        # Sort data by year
        # data.sort_values(by="year", inplace=True)

        # Split data
        train_data = data.iloc[: int(split_rate * len(data))]
        test_data = data.iloc[int(split_rate * len(data)) :]

        return train_data, test_data


if __name__ == "__main__":

    # Set up logging
    setup_logging()

    FEATURES = [
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

    # Load czech data
    czech_data_loader = StateDataLoader("Czechia")
    czech_data = czech_data_loader.load_data()

    train, test = czech_data_loader.split_data(czech_data)

    # TODO: fix this
    scaled_data, scaler = czech_data_loader.scale_data(
        data=czech_data, features=FEATURES, scaler=MinMaxScaler()
    )

    print("-" * 100)
    print("Train data tail:")
    print(train.tail())

    print("-" * 100)
    print("Test data head:")
    print(test.head())

    # Preprocess data
    print("-" * 100)
    print("Preprocess data:")
    input_sequences, target_sequences = czech_data_loader.preprocess_training_data(
        train, 5, ["year", "population, total"]
    )

    input_batches, target_batches = czech_data_loader.create_batches(
        6, input_sequences=input_sequences, target_sequences=target_sequences
    )

    print("-" * 100)
    print("Batches: ")

    # Input batches
    print("-" * 100)
    print("Input:")
    print(input_batches.shape)

    # Output batches
    print("-" * 100)
    print("Target:")
    print(target_batches.shape)
