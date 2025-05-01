import numpy as np
import pandas as pd
from typing import Dict, List, Union
from statsmodels.tsa.stattools import adfuller

from typing import List


import pandas as pd
from src.utils.constants import (
    basic_features,
    highly_correlated_features,
    categorical_columns,
)

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.data_transformer import DataTransformer

# print(basic_features(exclude=highly_correlated_features()))

from config import Config


settings = Config()


loader = StatesDataLoader()

STATE = "United States"
df_dict = loader.load_states(states=[STATE])

df = df_dict[STATE]
df.drop(columns=categorical_columns(), inplace=True)

transformer = DataTransformer()
transformed_df = transformer.transform_data(data=df, columns=df.columns)


class StationaryPreprocessor:

    def __init__(self):
        self.state_initial_values: Dict[str, Dict[str, float]] = {}

    # ADF Test
    def is_stationary_adf_test(self, series: pd.Series, alpha: float = 0.05) -> bool:
        """
        Check if the series might be stationary of using adfuller test.

        Args:
            series (pd.Series): Teste series.

        Returns:
            out: bool: Returns True if the series is stationary.
        """
        result = adfuller(series.dropna())
        return result[1] < alpha

    def get_non_stationary_columns_for_state(
        self, state: pd.DataFrame
    ) -> Dict[str, Union[int | float]]:

        # Apply the ADF test to the original data - for the state
        non_stationary_columns_base_values: Dict[str, Union[int | float]] = {}

        for column in df.columns:

            # If the series is not stationary save the column name with its initial value
            if not self.is_stationary_adf_test(state[column], column):

                initial_value = state.iloc[0][column]
                non_stationary_columns_base_values[column] = initial_value

        return non_stationary_columns_base_values

    def transform(self, state: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data by differencing non-stationary columns.

        Args:
            state (str): Identifier for the data group (e.g., state name).
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with non-stationary columns differenced.
        """
        self.state_initial_values[state] = self.get_non_stationary_columns_for_state(
            data
        )
        transformed = data.copy()

        for col in self.state_initial_values[state]:
            transformed[col] = transformed[col].diff()

        return transformed.dropna()

    def inverse_transform(
        self, state: str, transformed_data: pd.DataFrame
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
