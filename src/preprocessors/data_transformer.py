# Standard library imports
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Callable, Union, Optional
from sklearn.preprocessing import MinMaxScaler
import warnings

import torch
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.stattools import adfuller

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader
from src.utils.constants import (
    get_core_hyperparameters,
    aging_targets,
    basic_features,
    highly_correlated_features,
)


# TODO:
# 1. transform data first
# 2. split and scale them afterwards


class DataTransformer:

    # Categorical columns
    CATEGORICAL_COLUMNS: List[str] = ["country_name"]

    # Division of features by types
    ABSOLUTE_COLUMNS: List[str] = [
        # Features
        "year",
        "fertility_rate_total",
        "birth_rate_crude",
        "adolescent_fertility_rate",
        "death_rate_crude",
        "life_expectancy_at_birth_total",
    ]

    PERCENTUAL_COLUMNS: List[str] = [
        # Features
        "population_growth",
        "arable_land",
        "gdp_growth",
        "agricultural_land",
        "rural_population",
        "rural_population_growth",
        "urban_population",
        "age_dependency_ratio",
        # Targets
        "population_ages_15-64",
        "population_ages_0-14",
        "population_ages_65_and_above",
        "population_female",
        "population_male",
    ]

    WIDE_RANGE_COLUMNS: Dict[str, Callable] = {
        "net_migration": "transform_wide_range_data",
        "population_total": "transform_wide_range_data",
        "gdp": "transform_wide_range_data",
    }

    def __init__(self):

        # Save scalers for features and targets separately
        self.SCALER: MinMaxScaler | None = None
        self.TARGET_SCALER: MinMaxScaler | None = None

        # Get fitted label encoders per columns
        self.LABEL_ENCODERS: Dict[str, LabelEncoder] | None = None

        # Initial values for states used for differentiaion
        # Format {"Uganda": {"gdp": 1200425, "population_total": 1234925}, "Czechia": ...}
        self.state_initial_values: Dict[str, Dict[str, Union[int, float]]] = {}

    def transform_categorical_columns(
        self,
        data: pd.DataFrame,
        inverse: bool = False,
    ):

        # Copy dataframe to avoid direct modification
        to_encode_data = data.copy()

        # Filter out supported categorical columns
        categorical_columns = list(set(self.CATEGORICAL_COLUMNS) & set(data.columns))
        to_encode_data = to_encode_data[categorical_columns]

        # Create new label encoders if the label encoders is None and there are some columns
        if self.LABEL_ENCODERS is None and categorical_columns:
            new_label_encoders: Dict[str, LabelEncoder] = {}
            for col in categorical_columns:
                le = LabelEncoder()
                to_encode_data[col] = le.fit_transform(to_encode_data[col])
                new_label_encoders[col] = (
                    le  # Save encoders if you want to inverse later
                )

            # Save the encoders dict
            self.LABEL_ENCODERS = new_label_encoders
            return to_encode_data

        # If the label encoders are created
        for col in categorical_columns:

            # For inverse transformation
            if inverse:
                to_encode_data[col] = self.LABEL_ENCODERS[col].inverse_transform(
                    to_encode_data[col]
                )
            else:
                to_encode_data[col] = self.LABEL_ENCODERS[col].transform(
                    to_encode_data[col]
                )

        return to_encode_data[categorical_columns]

    def transform_percentual_columns(
        self, data: pd.DataFrame, inverse: bool = False
    ) -> pd.DataFrame:

        to_scale_data = data.copy()

        # Get available percentual columns
        percentual_columns = list(set(self.PERCENTUAL_COLUMNS) & set(data.columns))

        if inverse:
            return to_scale_data[percentual_columns].apply(lambda x: x * 100)

        return to_scale_data[percentual_columns].apply(lambda x: x / 100)

    def transform_wide_range_data(
        self, data: pd.DataFrame, inverse: bool = False, C: float = 1.0
    ) -> pd.DataFrame:

        to_scale_data = data.copy()

        # Decode
        if inverse:
            # return to_scale_data.apply(lambda col: np.sign(col) * np.expm1(np.abs(col)))
            return to_scale_data.apply(
                lambda col: np.sign(col) * C * (-1 + 10 ** (np.abs(col) / C))
            )
        # Encode
        # return to_scale_data.apply(lambda col: np.sign(col) * np.log1p(np.abs(col)))
        return to_scale_data.apply(
            lambda col: np.sign(col) * (np.log10(1 + np.abs(col) / C))
        )

    def is_stationary_adf_test(
        self, series: pd.Series, col: str, alpha: float = 0.05
    ) -> bool:
        """
        Perform the Augmented Dickey-Fuller test to determine stationarity.

        Args:
            series (pd.Series): The time series.
            alpha (float): Significance level (default: 0.05).

        Returns:
            bool: True if the series is stationary, False otherwise.
        """

        # Check for constant series - if the series is contant, automatically detected as non-stationary
        # Check for constant or too short series

        series = series.dropna()
        if series.nunique() <= 1:
            return False
        if len(series) < 10:
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = adfuller(series)
            p_value = result[1]

            # If SSR is zero, adfuller may still return a p-value of 0.0 — we can catch this:
            if np.isinf(result[0]) or np.isnan(p_value):
                return False

            return p_value < alpha
        except Exception as e:
            print(f"ADF test failed on series: {e}")
            return False

    def identify_non_stationary_columns(
        self, state_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Identify non-stationary columns in a given state's DataFrame.

        Args:
            state_df (pd.DataFrame): Time series data for one state.

        Returns:
            Dict[str, float]: Columns that are non-stationary with their initial values.
        """
        non_stationary = {}
        for column in state_df.columns:
            if not self.is_stationary_adf_test(state_df[column], col=column):
                non_stationary[column] = state_df[column].iloc[0]

        return non_stationary

    def differentiate(self, state: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data by differencing non-stationary columns.

        Args:
            state (str): Identifier for the data group (e.g., state name).
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with non-stationary columns differenced.
        """
        self.state_initial_values[state] = self.identify_non_stationary_columns(data)
        transformed = data.copy()

        for col in self.state_initial_values[state]:
            transformed[col] = transformed[col].diff()

        return transformed.dropna()

    def inverse_differentiate(
        self,
        state: str,
        transformed_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inverts the differencing transformation to recover the original data.

        Args:
            state (str): Identifier for the data group.
            transformed_data (pd.DataFrame): Differenced data.

        Returns:
            pd.DataFrame: Reconstructed data.
        """
        if state not in self.state_initial_values:
            raise ValueError(f"No initial values stored for state '{state}'")

        reconstructed = transformed_data.copy()
        for col, init_val in self.state_initial_values[state].items():

            reconstructed[col] = reconstructed[col].cumsum() + init_val

        return reconstructed

    # I can use this per state and then merge it to one big dataset
    def transform_data(
        self,
        state: str,
        data: pd.DataFrame,
        columns: List[str],
        inverse: bool = False,
    ) -> pd.DataFrame:

        # If series is too short for ADF test
        if len(data) < 10 and state not in self.state_initial_values.keys():
            print(f"Series too short for ADF test (length={len(data)}). Skipping.")
            return pd.DataFrame()

        # Transforms data using specified transformation
        to_transform_data = data.copy()

        ORIGINAL_DATA = data.copy()
        ORIGINAL_COLUMNS = data.columns

        # if inverse:
        #     to_transform_data = self.inverse_differentiate(
        #         state=state,
        #         transformed_data=to_transform_data,
        #     )

        # Process categorical columns
        categorical_df = self.transform_categorical_columns(
            data=to_transform_data[columns], inverse=inverse
        )

        # Process percentual columns - normalize from 0 - 100 to range 0 - 1
        percentual_df = self.transform_percentual_columns(
            data=to_transform_data[columns], inverse=inverse
        )

        # Get available absolute columns - leave as is, just use min max scaling
        absolute_columns = list(set(self.ABSOLUTE_COLUMNS) & set(columns))
        absolute_columns_df = to_transform_data[absolute_columns]

        # # Process special columns
        # wide_range_columns = list(set(self.WIDE_RANGE_COLUMNS) & set(columns))
        # wide_range_columns_df = self.transform_wide_range_data(
        #     data=to_transform_data[wide_range_columns]
        # )

        # TODO: Maybe get rid of this
        WIDE_RANGE_COLUMNS_dfs: List[pd.DataFrame] = []
        for col, func_name in self.WIDE_RANGE_COLUMNS.items():
            if col in columns:

                # Dynamically call the transformation methods
                transform_func = getattr(self, func_name)
                WIDE_RANGE_COLUMNS_dfs.append(
                    transform_func(to_transform_data[col], inverse=inverse)
                )

        # Merge transformed columns
        transformed_data_df = pd.concat(
            [
                categorical_df,
                absolute_columns_df,
                percentual_df,
                *WIDE_RANGE_COLUMNS_dfs,
            ],
            axis=1,
        )

        # Ensure stationarity
        if not inverse:
            transformed_data_df = self.differentiate(
                state=state, data=transformed_data_df
            )

        # Reconstruct the original dataframes
        non_transformed_features = [f for f in ORIGINAL_COLUMNS if not f in columns]

        # Get the t
        transformed_original_data = pd.concat(
            [
                ORIGINAL_DATA[non_transformed_features].reset_index(drop=True),
                transformed_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        # print(transformed_original_data[FEATURES])
        return transformed_original_data[ORIGINAL_COLUMNS].dropna()

    def __scale_and_fit(
        self,
        training_data: pd.DataFrame,
        columns: List[str],
        scaler: MinMaxScaler,
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        This method is scaling for the model training. Fit specified scaler on the training data.

        Args:
            training_data (Dict[str, pd.DataFrame]): Training data dict. The key is the state name.
            columns (List[str]): The columns for scaling.
            scaler (MinMaxScaler): Scaler to be fitted on training data.

        Returns:
            Tuple[pd.DataFrame, MinMaxScaler]: scaled_data, fitted_scaler
        """
        # Transforms raw data, fits the given scaler
        # Should be used on training_data
        # Get the original columns of the first state, should be all the same
        ORIGINAL_COLUMNS = training_data.columns

        # scaled_states_data_dict: Dict[str, pd.DataFrame] = {}
        # for state, df in training_data.items():
        ORIGINAL_DATA_TRAINING_DATA = training_data.copy()

        # Scale and fit data
        scaled_data = scaler.fit_transform(training_data[columns])

        scaled_training_data_df = pd.DataFrame(scaled_data, columns=columns)

        # Reconstruct the original dataframes
        non_transformed_features = [f for f in ORIGINAL_COLUMNS if not f in columns]

        # Get the t
        scaled_training_data_df = pd.concat(
            [
                ORIGINAL_DATA_TRAINING_DATA[non_transformed_features].reset_index(
                    drop=True
                ),
                scaled_training_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        # scaled_states_data_dict[state] = scaled_training_data_df

        return (
            scaled_training_data_df[ORIGINAL_COLUMNS],
            scaler,
        )

    def scale_and_fit(
        self,
        training_data: pd.DataFrame,
        features: List[str],
        targets: List[str] | None = None,
    ) -> pd.DataFrame:
        """
        This method is scaling for the model training. Fit specified scaler on the training data.

        Args:
            training_data (pd.DataFrame): Training data.
            columns (List[str]): The columns for scaling.
            scaler (MinMaxScaler): Scaler to be fitted on training data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: scaled_feature_data, scaled_target_data
        """

        FEATURES: List[str] = features
        if targets:
            FEATURES: List[str] = [f for f in features if not f in targets]
            TARGETS: List[str] = targets

        # Scale and fit feature data

        scaled_training_feature_data, feature_scaler = self.__scale_and_fit(
            training_data=training_data,
            columns=FEATURES,
            scaler=MinMaxScaler(),
        )

        # Scale and fit targets
        if targets:

            scaled_training_data, target_scaler = self.__scale_and_fit(
                training_data=scaled_training_feature_data,
                columns=TARGETS,
                scaler=MinMaxScaler(),
            )
            self.TARGET_SCALER = target_scaler
        else:
            scaled_training_data = scaled_training_feature_data

        # Save scalers
        self.SCALER = feature_scaler

        return scaled_training_data

    def scale_data(
        self,
        data: pd.DataFrame,
        features: List[str],
        targets: List[str] | None = None,
    ) -> pd.DataFrame:

        if self.SCALER is None:
            raise ValueError(
                "The scaler isnt fitted yet. Please use scale_and_fit first."
            )

        to_scale_data = data.copy()
        ORIGINAL_COLUMNS = data.columns

        # Maintain the original column order
        FEATURES = features
        TARGETS = targets

        if not targets:
            TARGETS = []

        # Check if features are equal to targets. This is just for ensure the compatibility with univariate neural network models.
        if FEATURES != TARGETS:
            FEATURES = [f for f in features if not f in TARGETS]

        to_scale_feature_data = to_scale_data[FEATURES]

        scaled_feature_data = self.SCALER.transform(to_scale_feature_data)
        scaled_feature_data_df = pd.DataFrame(scaled_feature_data, columns=FEATURES)

        scaled_data_df = scaled_feature_data_df

        if TARGETS:

            to_scale_target_data = to_scale_data[TARGETS]

            scaled_target_data = self.TARGET_SCALER.transform(to_scale_target_data)
            scaled_target_data_df = pd.DataFrame(scaled_target_data, columns=TARGETS)

            # Update scaled data if there is also a target scaling
            scaled_data_df = pd.concat([scaled_data_df, scaled_target_data_df], axis=1)

        # Reconstruct the original dataframe
        non_transformed_features = [
            f for f in ORIGINAL_COLUMNS if not f in FEATURES and f not in TARGETS
        ]

        scaled_data_df = pd.concat(
            [
                data[non_transformed_features].reset_index(drop=True),
                scaled_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return scaled_data_df[ORIGINAL_COLUMNS]

    def unscale_data(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
    ) -> pd.DataFrame:

        if self.TARGET_SCALER is None and self.SCALER is None:
            raise ValueError("No scaler is fitted yet. Please use scale_and_fit first.")

        # Check the fitted columns and specified columns compatibility?
        to_unscale_data = data.copy()

        ORIGINAL_COLUMNS = data.columns

        # Maintain the original column order
        FEATURES = features
        TARGETS = targets

        # Init unsacled dfs
        unscaled_feature_data_df = None
        unscaled_target_data_df = None

        # Unscale features
        if FEATURES:
            unscaled_feature_data = self.SCALER.inverse_transform(
                to_unscale_data[FEATURES]
            )
            unscaled_feature_data_df = pd.DataFrame(
                unscaled_feature_data, columns=FEATURES
            )

        # Unscale targets
        if TARGETS:

            if self.TARGET_SCALER:
                unscaled_target_data = self.TARGET_SCALER.inverse_transform(
                    to_unscale_data[TARGETS]
                )
                unscaled_target_data_df = pd.DataFrame(
                    unscaled_target_data, columns=TARGETS
                )

        # Merge unscaled data
        if not unscaled_feature_data_df is None and not unscaled_target_data_df is None:
            unscaled_data_df = pd.concat(
                [unscaled_feature_data_df, unscaled_target_data_df], axis=1
            )
        elif not unscaled_target_data_df is None:
            unscaled_data_df = unscaled_target_data_df

        elif not unscaled_feature_data_df is None:
            unscaled_data_df = unscaled_feature_data_df
        else:
            raise ValueError("Features or targets not specified.")

        INTEGER_COLUMNS = ["year"]  # Add more if needed

        for col in INTEGER_COLUMNS:
            if col in unscaled_data_df.columns:
                unscaled_data_df[col] = unscaled_data_df[col].round().astype(int)

        # Reconstruct the original dataframe

        TRANSFORMED_COLUMNS = (TARGETS if TARGETS else []) + (
            FEATURES if FEATURES else []
        )
        non_transformed_features = [
            f for f in ORIGINAL_COLUMNS if f not in TRANSFORMED_COLUMNS
        ]

        unscaled_data_df = pd.concat(
            [
                data[non_transformed_features].reset_index(drop=True),
                unscaled_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return unscaled_data_df[ORIGINAL_COLUMNS]

    # TODO:
    def scale_and_transform(self):
        pass

    def unscale_and_inverse_transform(self):
        pass


def main():
    STATE = "Czechia"

    # Load data
    loader = StateDataLoader(state=STATE)
    transformer = DataTransformer()

    data = loader.load_data()

    # EXCLUDE_COLUMNS = ["country_name", "year"]
    FEATURES = basic_features(exclude=highly_correlated_features())
    TARGETS = aging_targets()

    # Split data
    train_df, test_df = loader.split_data(data=data)

    print("Original train data:")
    print(train_df.iloc[1:][FEATURES].head())
    print()

    transformed_training_data = transformer.transform_data(
        state=STATE, data=train_df, columns=FEATURES
    )

    print("Transformed training data:")
    print(transformed_training_data.head()[FEATURES])
    print()

    # Scale training data
    scaled_trainig_data = transformer.scale_and_fit(
        training_data=transformed_training_data,
        features=FEATURES,
        targets=TARGETS,
    )

    print("Scaled train data:")
    print(scaled_trainig_data.head()[FEATURES])
    print()

    # Unscale data
    unscaled_training_data = transformer.unscale_data(
        data=scaled_trainig_data,
        features=FEATURES,
    )

    print("Unscaled train data:")
    print(unscaled_training_data.head()[FEATURES])
    print()

    reverse_transformed_training_data = transformer.transform_data(
        state=STATE, data=unscaled_training_data, columns=FEATURES, inverse=True
    )

    print("Reverse transformed training data:")
    print(reverse_transformed_training_data.head()[FEATURES])
    print()

    print()
    print("-" * 100)
    print()

    exit()

    # Scale test data using fitted scaler
    print("Original test data:")
    print(test_df.head()[TARGETS])
    print()

    scaled_test_data = transformer.scale_data(
        data=test_df, features=FEATURES, targets=TARGETS
    )
    print("Scaled test data:")
    print(scaled_test_data.head()[TARGETS])
    print()

    unscaled_test_data = transformer.unscale_data(
        data=scaled_test_data, targets=TARGETS
    )
    print("Unscaled test data:")
    print(unscaled_test_data.head()[TARGETS])
    print()


if __name__ == "__main__":
    main()
