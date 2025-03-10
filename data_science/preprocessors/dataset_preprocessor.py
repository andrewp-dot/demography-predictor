import os
import pandas as pd
import numpy as np
import pprint

from scipy.interpolate import interp1d

from typing import List, Dict, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import DatasetCreatorSettings
from base import BasePreprocessor

settings = DatasetCreatorSettings()


MAX_NAN_VALUES = 0


class DatasetPreprocessor(BasePreprocessor):

    def __init__(self, source_data_dir: str, to_save_data_dir: str):
        super().__init__(to_save_data_dir)
        self.source_data_dir = source_data_dir
        self.SCALER: MinMaxScaler | RobustScaler | StandardScaler = StandardScaler()

    def split_training_validation_dataset(
        self, df: pd.DataFrame, division_rate: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads states data and divides it to training and validation dataset.

        Args:
            division_rate (float, optional): Rate which selects the division rate of states.
            For example, rate = 0.2 means is equal to 10 states used for validation dataset from 50. Defaults to 0.2.
            preprocess (bool, optional): Set if the data should be preprocessed. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: training dataset, validation dataset
        """

        row_count = df.shape[0]

        # secure division rate
        division_rate = max(0, min(1, division_rate))

        # get the number of dataset files
        training_number = int(row_count * (1 - division_rate))

        # craft training dataset
        training_df = df.iloc[:training_number]

        # craft training dataset
        validation_df = df.iloc[training_number:]

        return training_df, validation_df

    def get_csv_paths(self):
        import re

        return [
            os.path.join(self.source_data_dir, file)
            for file in os.listdir(self.source_data_dir)
            if re.match(r"(?!Metadata).*[.]csv", file)
        ]

    def find_non_unnamend_cols(self, df: pd.DataFrame) -> List[str]:
        non_unnamed_columns = [col for col in df.columns if "unnamed" in col.lower()]
        return non_unnamed_columns

    def melt_single_df(
        self,
        df: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> pd.DataFrame:

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
        attribute_dfs = []

        for path in csv_paths:
            print(path)

            new_df = pd.read_csv(path, skiprows=skip_rows)

            # drop unnamed for all cases
            unnamed_columns = self.find_non_unnamend_cols(new_df)

            if unnamed_columns:
                new_df.drop(columns=unnamed_columns, inplace=True)

            # melt years
            new_df = self.melt_single_df(
                new_df,
                columns_to_keep=[
                    "Country Name",
                    "Country Code",
                    "Indicator Name",
                    "Indicator Code",
                ],
            )

            # drop useless columns
            new_df.drop(columns=["Indicator Name", "Indicator Code"], inplace=True)
            attribute_dfs.append(new_df)

        # merge data based on columns
        merged_df = attribute_dfs[0]

        for df in attribute_dfs[1:]:

            merged_df = pd.merge(
                merged_df,
                df,
                on=["Country Name", "Country Code", "Year"],
                how="inner",  # You can change this to 'left', 'right', or 'outer' depending on requirements
            )

        merged_df["NaN_count"] = merged_df.isna().sum(axis=1)

        return merged_df

    def interpolate_state(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO:

        numeric_cols_df = df.select_dtypes(include=[np.number])  # Only numeric columns
        non_numeric_cols_df = df.select_dtypes(
            exclude=[np.number]
        )  # Non-numeric columns

        # Interpolate data in columns, where you can interpolate them
        def spline_interpolate_column(series: pd.Series):

            # identify non-missing values
            x = series[series.notna()].index  # Known indices
            y = series[series.notna()].values  # Known values

            missing = series[series.isna()].index  # Indices with missing values
            if len(missing) == 0 or len(series) == 0:
                return series

            # if there are enough points, interpolate
            if (
                len(x) / len(series) > 0.7 and len(x) > 2
            ):  # for cubic interpolation there must by minimum of 2 values
                spline = interp1d(x, y, kind="cubic", fill_value="extrapolate")
                series[missing] = spline(missing)
            return series

        # Apply interpolation to each column
        interpolated_numeric_df = numeric_cols_df.apply(spline_interpolate_column)

        interpolated_df = pd.concat(
            [interpolated_numeric_df, non_numeric_cols_df], axis=1
        )

        # restore the column order
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

        # TODO:
        # 1. extract states
        states = df["Country Name"].unique()

        print(f"[Interpolate]: {df.shape}")

        state_dfs = []
        for state in states:
            state_df = df[df["Country Name"] == state]
            state_dfs.append(state_df)

        # 2. for every column in every state interpolate missing values from last known min value to last known max value
        interpolated_states = []
        for state_df in state_dfs:
            interpolated_state = self.interpolate_state(state_df)
            interpolated_states.append(interpolated_state)

        # 3. concat states
        interpolated_df = pd.concat(interpolated_states, axis=0)

        # 4. return interpolated
        return interpolated_df

    def clean_data(self, df: pd.DataFrame, min_seq_length: int) -> pd.DataFrame:

        # remove population with less than X years of records
        country_info_dict = country_sequences_info(df)

        df_copy = df.copy()

        for state, info in country_info_dict.items():
            min_year, max_year, missing_years = info

            state_development_years = max_year - min_year
            # if the state does not have a required length of the data
            # or there are more missing years than the actual values
            if state_development_years < min_seq_length or missing_years:
                df_copy = df_copy[df_copy["Country Name"] != state]

        # drop columns
        df_copy = df_copy[df_copy["NaN_count"] <= MAX_NAN_VALUES]
        df_copy.drop(columns=["Country Code"], inplace=True)
        df_copy.drop(columns=["NaN_count"], inplace=True)
        return df_copy

    def scale_data(
        self,
        df: pd.DataFrame,
        exlcude_columns: List[str],
        scaler: MinMaxScaler | StandardScaler | RobustScaler,
    ) -> pd.DataFrame:
        current_scaler: MinMaxScaler | StandardScaler | RobustScaler = scaler()

        not_scaled_df: pd.DataFrame = df[exlcude_columns]

        desired_columns: List[str] = [
            col for col in df.columns if col not in exlcude_columns
        ]

        scaled_data = current_scaler.fit_transform(df[desired_columns])

        scaled_df = pd.DataFrame(scaled_data, columns=desired_columns, index=df.index)

        # TODO: more description to the scaler
        self.SCALER = current_scaler

        # concat columns
        return pd.concat([not_scaled_df, scaled_df], axis=1)

    def process(
        self,
    ) -> pd.DataFrame:
        """Processes states data. This is specified preprocessor just for state data configured in 'config.py' in the root of the project.

        Args:
            name (str, optional): Name of the processed dataset file name. Defaults to "preprocessed_states.csv".

        Returns:
            Dataframe: processed dataset
        """

        # get csv file paths
        csv_paths = self.get_csv_paths()

        skip_rows = 4
        merged_df = self.merge_indicators(csv_paths=csv_paths, skip_rows=skip_rows)

        # TODO: sort values
        merged_df = merged_df.sort_values(by=["Country Name", "Year"])

        print(f"[Process]: {merged_df.shape}")

        merged_df = self.interpolate(merged_df)

        print(f"[Process Interpolated]: {merged_df.shape}")

        # TODO: clean_data
        merged_df = self.clean_data(merged_df, 10)

        # TODO: remove non-periodic sequences

        return merged_df

    def save_states_separately(self, df: pd.DataFrame, dir_name: str):
        # get name without csv

        dir_name = dir_name.replace(".csv", "")
        states_dir = os.path.join(self._to_save_data_dir, "states")

        # make dir if not present
        if not os.path.isdir(states_dir):
            os.makedirs(states_dir, exist_ok=True)

        # get states and save used states data
        states = df["Country Name"].unique()

        for state in list(states):
            state_df = df[df["Country Name"] == state]

            state_file_name = os.path.join(states_dir, f"{state}.csv")
            state_df.to_csv(state_file_name, index=False)


def find_missing_years(df: pd.DataFrame, min_year: int, max_year: int) -> List[int]:

    missing_years = []
    for y in range(min_year, max_year):
        if y not in list(df["Year"].unique()):
            missing_years.append(y)

    return missing_years


def country_sequences_info(df: pd.DataFrame) -> dict[str, tuple[int, int, list]]:

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


if __name__ == "__main__":
    source_data_dir = settings.source_data_dir

    to_save_dir = os.path.join(
        settings.save_dataset_path, "datasets", f"dataset_{settings.dataset_version}"
    )

    if not os.path.exists(to_save_dir):
        os.makedirs(to_save_dir)

    # define scaler
    SCALER = RobustScaler

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

    # TODO: interpolate populations with years missing, complete sequences

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

    # rename columns
    import re

    original_cols = list(merged.columns)
    mapping_dict = {
        original: re.sub(r"\(.*?\)", "", original).strip() for original in original_cols
    }

    print(mapping_dict)

    print("Renaming columns ... ")
    merged.rename(mapping_dict, inplace=True, axis=1)
    merged.columns = merged.columns.str.replace('"', "", regex=False)

    print(f"After after processing: {merged.shape}")
    preprocessor.save_states_separately(
        merged, f"dataset_{settings.dataset_version}_states"
    )

    preprocessor.save_data(merged, f"dataset_{settings.dataset_version}.csv")

    print(merged.describe())

    # scale data
    # merged = preprocessor.scale_data(
    #     df=merged,
    #     exlcude_columns=["Country Name"],
    #     scaler=SCALER,
    # )

    print(merged.head())
