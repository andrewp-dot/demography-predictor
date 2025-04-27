import os
import pandas as pd
import numpy as np
import pprint

from scipy.interpolate import interp1d

from typing import List, Dict, Tuple, Union

from config import DatasetCreatorSettings
from data_science.preprocessors.base import BasePreprocessor

settings = DatasetCreatorSettings()

# This is main in here
# TODO:
# 1. Comments in here -> make code more readable
# 2. Delete states with low number of records

MAX_NAN_VALUES = 0


class DatasetPreprocessor(BasePreprocessor):

    def __init__(self, source_data_dir: str, to_save_data_dir: str):
        super().__init__(to_save_data_dir)
        self.source_data_dir = source_data_dir

    def get_csv_paths(self) -> List[str]:
        """
        Get all csv files from the source data directory.

        Returns:
            out: List[str]: List of path to csv file data sources from data.worldbank.
        """
        import re

        return [
            os.path.join(self.source_data_dir, file)
            for file in os.listdir(self.source_data_dir)
            if re.match(r"(?!Metadata).*[.]csv", file)
        ]

    def find_non_unnamend_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Get all columns that are not named 'unnamed'.

        Args:
            df (pd.DataFrame): Data frame to search for non-unnamed columns.

        Returns:
            out: List[str]: List of non-unnamed columns.
        """
        non_unnamed_columns = [col for col in df.columns if "unnamed" in col.lower()]
        return non_unnamed_columns

    def melt_single_df(
        self,
        df: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> pd.DataFrame:
        """
        Convert wide format to long format.

        Returns:
            out: pd.DataFrame: Dataframe in the long format.
        """

        long_format = pd.melt(
            df,
            id_vars=columns_to_keep,
            var_name="Year",  # Name of the new column for years
            value_name="Value",  # Name of the new column for values
        )

        long_format.rename(
            columns={"Value": long_format["Indicator Name"].iloc[0]}, inplace=True
        )
        # Convert the "Year" column to integer if needed
        long_format["Year"] = long_format["Year"].astype(int)

        return long_format

    def merge_indicators(self, csv_paths: List[str], skip_rows: int) -> pd.DataFrame:
        """
        Merges all indicators from the csv files. One csv files contains one indicator.

        Args:
            csv_paths (List[str]): _description_
            skip_rows (int): Skip first n rows in the csv file. (for example in order to skip metadata).

        Returns:
            out: pd.DataFrame: Merged indicators data frame.
        """

        attribute_dfs = []

        for path in csv_paths:
            print(f"[Merge indicators]: {path}")

            new_df = pd.read_csv(path, skiprows=skip_rows)

            # Drop unnamed for all cases
            unnamed_columns = self.find_non_unnamend_cols(new_df)

            if unnamed_columns:
                new_df.drop(columns=unnamed_columns, inplace=True)

            # Melt years
            new_df = self.melt_single_df(
                new_df,
                columns_to_keep=[
                    "Country Name",
                    "Country Code",
                    "Indicator Name",
                    "Indicator Code",
                ],
            )

            # Drop useless columns
            new_df.drop(columns=["Indicator Name", "Indicator Code"], inplace=True)
            attribute_dfs.append(new_df)

        # Merge data based on columns
        merged_df = attribute_dfs[0]

        for df in attribute_dfs[1:]:

            merged_df = pd.merge(
                merged_df,
                df,
                on=["Country Name", "Country Code", "Year"],
                how="inner",
            )

        # Get the number of NaN values in each row
        merged_df["NaN_count"] = merged_df.isna().sum(axis=1)

        return merged_df

    def interpolate_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use data interpolation to fill missing values in the state data frame. Interpolation is done for numeric columns only and for each column separately.

        Args:
            df (pd.DataFrame): Data frame to interpolate.

        Returns:
            out: pd.DataFrame: Data frame with interpolated values.
        """

        # Separate numeric and non-numeric columns
        numeric_cols_df = df.select_dtypes(include=[np.number])  # Only numeric columns
        non_numeric_cols_df = df.select_dtypes(
            exclude=[np.number]
        )  # Non-numeric columns

        # Interpolate data in columns, where you can interpolate them
        def spline_interpolate_column(series: pd.Series):

            # Identify non-missing values
            x = series[series.notna()].index  # Known indices
            y = series[series.notna()].values  # Known values

            # Indices with missing values
            missing = series[series.isna()].index
            if len(missing) == 0 or len(series) == 0:
                return series

            # If there are enough known points (more than 70% of values), interpolate
            if (
                len(y) / len(series) > 0.7 and len(x) > 2
            ):  # for cubic interpolation there must by minimum of 2 values
                spline = interp1d(x, y, kind="cubic", fill_value="extrapolate")
                series[missing] = spline(missing)
            return series

        # Apply interpolation to each column
        interpolated_numeric_df = numeric_cols_df.apply(spline_interpolate_column)

        interpolated_df = pd.concat(
            [interpolated_numeric_df, non_numeric_cols_df], axis=1
        )

        # Restore the column order
        interpolated_df = interpolated_df[df.columns]

        if interpolated_df.isna().sum().sum() < df.isna().sum().sum():
            print(
                f"{df.iloc[0]['Country Name']} NaN values before: {df.isna().sum().sum()}"
            )
            print(
                f"{interpolated_df.iloc[0]['Country Name']} NaN values after: {interpolated_df.isna().sum().sum()}"
            )

        return interpolated_df

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing data in the data frame. Interpolation is done for each state and each column in the state separately.

        Args:
            df (pd.DataFrame): Data frame to interpolate.

        Returns:
            out: pd.DataFrame: Interpolated data.
        """

        # TODO:
        # Extract states
        states = df["Country Name"].unique()

        print(f"[Interpolate]: {df.shape}")

        state_dfs = []
        for state in states:
            state_df = df[df["Country Name"] == state]
            state_dfs.append(state_df)

        # For every column in every state interpolate missing values from last known min value to last known max value
        interpolated_states = []
        for state_df in state_dfs:
            interpolated_state = self.interpolate_state(state_df)
            interpolated_states.append(interpolated_state)

        # Concat states
        interpolated_df = pd.concat(interpolated_states, axis=0)

        # Return interpolated
        return interpolated_df

    def clean_data(self, df: pd.DataFrame, min_seq_length: int) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            min_seq_length (int): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # Rmove population with less than X years of records
        country_info_dict = country_sequences_info(df)

        df_copy = df.copy()

        # TODO: fix this down there
        for state, info in country_info_dict.items():
            min_year, max_year, missing_years = info

            # TODO: fix this... (maybe there is wrong way of the state_development_years calculation)
            state_development_years = max_year - min_year
            # if the state does not have a required length of the data
            # or there are more missing years than the actual values
            if state_development_years < min_seq_length or missing_years:
                df_copy = df_copy[df_copy["Country Name"] != state]

        # Drop columns with too many NaN values
        df_copy = df_copy[df_copy["NaN_count"] <= MAX_NAN_VALUES]
        df_copy.drop(columns=["Country Code"], inplace=True)
        df_copy.drop(columns=["NaN_count"], inplace=True)
        return df_copy

    def process(self, skip_rows: int = 4) -> pd.DataFrame:
        """
        Processes states data. This is specified preprocessor just for state data configured in 'config.py' in the root of the project.

        Args:
            name (str, optional): Name of the processed dataset file name. Defaults to "preprocessed_states.csv".

        Returns:
            out: Dataframe: processed dataset
        """

        # Get csv file paths
        csv_paths = self.get_csv_paths()

        # Skip header and merge indicators
        merged_df = self.merge_indicators(csv_paths=csv_paths, skip_rows=skip_rows)

        # Sort values by country name and year
        merged_df = merged_df.sort_values(by=["Country Name", "Year"])

        print(f"[Process]: {merged_df.shape}")

        merged_df = self.interpolate(merged_df)

        print(f"[Process Interpolated]: {merged_df.shape}")

        # TODO: clean_data
        merged_df = self.clean_data(merged_df, 10)

        # TODO: remove non-periodic sequences

        return merged_df

    def save_states_separately(self, df: pd.DataFrame, dir_name: str):
        """
        Creates and saves separate csv files for each state from the merged dataset.

        Args:
            df (pd.DataFrame): Dataset with all states.
            dir_name (str): Name of the directory where the state files will be saved.
        """
        # Get name without csv
        dir_name = dir_name.replace(".csv", "")
        states_dir = os.path.join(self._to_save_data_dir, "states")

        # Create dir if not present
        if not os.path.isdir(states_dir):
            os.makedirs(states_dir, exist_ok=True)

        # Get states and save used states data
        states = df["country_name"].unique()

        for state in list(states):
            state_df = df[df["country_name"] == state]

            state_file_name = os.path.join(states_dir, f"{state}.csv")
            state_df.to_csv(state_file_name, index=False)


def find_missing_years(df: pd.DataFrame, min_year: int, max_year: int) -> List[int]:
    """
    Finds missing years in the data frame.

    Args:
        df (pd.DataFrame): Dataframe to search for missing years.
        min_year (int): Minumum year in the data frame.
        max_year (int): Maximum year in the data frame.

    Returns:
        out: List[int]: List of missing years.
    """

    missing_years = []
    for y in range(min_year, max_year):
        if y not in list(df["Year"].unique()):
            missing_years.append(y)

    return missing_years


def country_sequences_info(df: pd.DataFrame) -> dict[str, tuple[int, int, list]]:
    """
    Get information about the state sequences.

    Args:
        df (pd.DataFrame): Dataframe with all states data.

    Returns:
        out: dict[str, tuple[int, int, list]]: Dictionary with state names as keys and tuple of min year, max year and missing years as values.
    """

    # get all state values
    states = df["Country Name"].unique()

    # display states with min, max and missing years of data
    state_years: dict[str, tuple] = {}
    for state in list(states):
        state_df = df.where(df["Country Name"] == state)

        min_y = int(state_df["Year"].min())
        max_y = int(state_df["Year"].max())
        missing_years = find_missing_years(state_df, min_y, max_y)
        state_years[state] = (min_y, max_y, missing_years)

    # filter populations with not empty rows
    longest_evolution_period_dict = {
        key: value
        for key, value in state_years.items()
        # filter for minimum years?
        # if not value[2]
    }

    return longest_evolution_period_dict


def main() -> None:
    """
    Main function for the dataset preprocessor.
    """

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--version", type=str, help="Version of the dataset", required=True
    )

    args = parser.parse_args()
    dataset_version = args.version

    # Directory where the source data are stored
    source_data_dir = settings.source_data_dir

    # Save created dataset to directory path
    to_save_dir = os.path.join(settings.save_dataset_path, f"dataset_{dataset_version}")

    # Create directory if does not exist
    if not os.path.exists(to_save_dir):
        os.makedirs(to_save_dir)

    # Preprocess raw data
    preprocessor = DatasetPreprocessor(
        to_save_data_dir=to_save_dir, source_data_dir=source_data_dir
    )
    merged = preprocessor.process()

    print(f"After processing: {merged.shape}")

    def get_states_with_incomplete_sequence(
        df: pd.DataFrame,
    ) -> dict[str, tuple[int, int, list]]:

        country_info_dict = country_sequences_info(df)

        states_with_missing_years = {
            key: value for key, value in country_info_dict.items() if value[2]
        }

        return states_with_missing_years

    # 1. locate state with incomplete sequence
    states_with_missing_years = get_states_with_incomplete_sequence(merged)

    # get state names
    incomplete_sequence_states = list(states_with_missing_years.keys())

    # 1a. divide incomplete sequence states to separate dataframe
    incomplete_sequence_state_dfs = []
    for state in incomplete_sequence_states:
        incomplete_sequence_state_dfs.append(merged[merged["Country Name"] == state])

    # 1b. remove states with incomplete sequence from the main dataframe
    merged = merged[~merged["Country Name"].isin(incomplete_sequence_states)]

    # 2. decide, if there is a good reason to interpolate missing years
    interpolated_state_dfs = []

    for state_df, state_info in zip(
        incomplete_sequence_state_dfs, states_with_missing_years.items()
    ):

        state, info = state_info
        min_year, max_year, missing_years = info

        population_development_time = max_year - min_year

        # if there is missing more then 20% of the whole recorded development, do not interpolate
        if len(missing_years) / population_development_time > 0.2:
            print(f"Not interpolated: {state}")
            continue

        # 3. if yes, add empty years of data to rows
        all_years = pd.DataFrame(
            {
                "Country Name": state_df.iloc[0]["Country Name"],
                "Year": range(min_year, max_year + 1),
            }
        )
        state_df = pd.merge(
            all_years, state_df, on=["Country Name", "Year"], how="left"
        )

        # 4. interpolate data
        numeric_cols = state_df.select_dtypes(include="number").columns
        state_df[numeric_cols] = state_df[numeric_cols].interpolate(method="linear")

        # Add the interpolated state DataFrame to the list
        interpolated_state_dfs.append(state_df)

    # 5. return reacreated dataset
    merged = pd.concat([merged] + interpolated_state_dfs, axis=0)

    states_with_missing_years = get_states_with_incomplete_sequence(merged)
    pprint.pprint(states_with_missing_years)

    # Rename columns
    import re

    original_cols = list(merged.columns)
    mapping_dict = {
        original: re.sub(r"\(.*?\)", "", original).strip().lower()
        for original in original_cols
    }

    print(mapping_dict)

    print("Renaming columns ... ")
    merged.rename(mapping_dict, inplace=True, axis=1)

    # Delete unicode characters because of lightgbm
    merged.columns = merged.columns.str.replace(r'["\'\\{}[\]:,]', "", regex=True)

    # Replace space with underscore
    merged.columns = merged.columns.str.replace(" ", "_")

    print(f"After after processing: {merged.shape}")
    preprocessor.save_states_separately(merged, f"dataset_{dataset_version}_states")

    preprocessor.save_data(merged, f"dataset_{dataset_version}.csv")

    print(merged.describe())

    print(merged.head())


if __name__ == "__main__":
    main()
