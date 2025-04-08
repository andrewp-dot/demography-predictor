# Standard library imports
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Callable
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader


# TODO:
# 1. Save this as preprocessor for data
# 2. Preprocess categorical data
# 3. Understand the data -> plot the dataset etc.


class DataTransformer:

    # Categorical columns
    CATEGORICAL_COLUMNS: List[str] = ["country name"]

    # Division of features by types
    ABSOLUTE_COLUMNS: List[str] = [
        # Features
        "year",
        "fertility rate, total",
        "birth rate, crude",
        "adolescent fertility rate",
        "death rate, crude",
        "life expectancy at birth, total",
    ]

    PERCENTUAL_COLUMNS: List[str] = [
        # Features
        "population growth",
        "arable land",
        "gdp growth",
        "agricultural land",
        "rural population",
        "rural population growth",
        "urban population",
        "age dependency ratio",
        # Targets
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
        "population, female",
        "population, male",
    ]

    SPECIAL_COLUMNS: Dict[str, Callable] = {
        "net migration": "transform_net_migration",
        "population, total": "transform_population_total",
    }

    def __init__(self):
        self.SCALER: MinMaxScaler | None = None

    def transform_categorical_columns(
        self,
        data: pd.DataFrame,
        inverse: bool = False,
    ):

        # TODO: use label encoder or something
        raise NotImplementedError("")

    def transform_percentual_columns(
        self, data: pd.DataFrame, inverse: bool = False
    ) -> pd.DataFrame:

        to_scale_data = data.copy()

        # Get available percentual columns
        percentual_columns = list(set(self.PERCENTUAL_COLUMNS) & set(data.columns))

        if inverse:
            return to_scale_data[percentual_columns].apply(lambda x: x * 100)

        return to_scale_data[percentual_columns].apply(lambda x: x / 100)

    def transform_net_migration(
        self, net_migration_data: pd.DataFrame, inverse: bool = False
    ) -> pd.DataFrame:
        to_scale_data = net_migration_data.copy()

        # TODO: maybe some sort of scaling?

        # Decode
        if inverse:
            return to_scale_data.apply(lambda col: np.sign(col) * np.expm1(np.abs(col)))
        # Encode
        return to_scale_data.apply(lambda col: np.sign(col) * np.log1p(np.abs(col)))

    def transform_population_total(
        self, population_total_data: pd.DataFrame, inverse: bool = False
    ):
        to_scale_data = population_total_data.copy()

        # Decode
        if inverse:
            return to_scale_data.apply(lambda col: np.sign(col) * np.expm1(np.abs(col)))
        # Encode
        return to_scale_data.apply(lambda col: np.sign(col) * np.log1p(np.abs(col)))

    def scale_data(
        self, data: pd.DataFrame, columns: List[str], scaler: MinMaxScaler | None = None
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:

        to_scale_data = data.copy()

        # Maintain the original column order
        FEATURES = columns

        # Process categorical columns
        # categorical_columns = set(CATEGORICAL_COLUMNS) & all_columns_set

        # Process percentual columns - normalize from 0 - 100 to range 0 - 1
        percentual_df = self.transform_percentual_columns(data=to_scale_data)

        # Get available absolute columns - leave as is, just use min ma scaling
        absolute_columns = list(set(self.ABSOLUTE_COLUMNS) & set(columns))
        absolute_columns_df = to_scale_data[absolute_columns]

        # Process special columns
        special_columns_dfs: List[pd.DataFrame] = []
        for col, func_name in self.SPECIAL_COLUMNS.items():
            if col in columns:

                # Dynamically call the transformation methods
                transform_func = getattr(self, func_name)
                special_columns_dfs.append(transform_func(to_scale_data[col]))

        # Merge transformed columns
        transformed_data_df = pd.concat(
            [absolute_columns_df, percentual_df, *special_columns_dfs], axis=1
        )

        # Maintain the column order
        transformed_data_df = transformed_data_df[FEATURES]

        # If there is no scaler specified, create new
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(transformed_data_df)

        # Use scaler
        scaled_data = scaler.transform(transformed_data_df)
        self.SCALER = scaler

        scaled_data_df = pd.DataFrame(scaled_data, columns=transformed_data_df.columns)

        return scaled_data_df, scaler

    def unscale_data(
        self, data: pd.DataFrame, columns: List[str], fitted_scaler: MinMaxScaler
    ) -> pd.DataFrame:

        # Check the fitted columns and specified columns compatibility?
        to_unscale_data = data.copy()

        # Maintain the original column order
        FEATURES = columns

        # Unsale data
        unscaled_data = fitted_scaler.inverse_transform(to_unscale_data)
        unscaled_data_df = pd.DataFrame(unscaled_data, columns=columns)

        # Inverse transform data

        # Process percentual columns - normalize from 0 - 100 to range 0 - 1
        percentual_df = self.transform_percentual_columns(
            data=unscaled_data_df, inverse=True
        )

        # Process absolute columns
        absolute_columns = list(set(self.ABSOLUTE_COLUMNS) & set(columns))
        absolute_columns_df = unscaled_data_df[absolute_columns]

        # Process special columns
        special_columns_dfs: List[pd.DataFrame] = []

        for col, func_name in self.SPECIAL_COLUMNS.items():
            if col in columns:
                # Dynamically call the transformation methods
                transform_func = getattr(self, func_name)
                special_columns_dfs.append(
                    transform_func(unscaled_data_df[col], inverse=True)
                )

        reverse_transformed_data_df = pd.concat(
            [absolute_columns_df, percentual_df, *special_columns_dfs], axis=1
        )

        INTEGER_COLUMNS = ["year"]  # Add more if needed

        for col in INTEGER_COLUMNS:
            if col in reverse_transformed_data_df.columns:
                reverse_transformed_data_df[col] = (
                    reverse_transformed_data_df[col].round().astype(int)
                )

        return reverse_transformed_data_df[FEATURES]


def main():
    STATE = "Czechia"

    # Load data
    loader = StateDataLoader(state=STATE)
    transformer = DataTransformer()

    data = loader.load_data()

    EXCLUDE_COLUMNS = ["country name"]
    COLUMNS = [col for col in data.columns if col not in EXCLUDE_COLUMNS]

    # Save original data
    # data[COLUMNS].to_csv(f"original_{STATE}_data.csv")
    print("Original data:")
    print(data[COLUMNS].head())
    print()

    # Scale data
    scaled_data, fitted_scaler = transformer.scale_data(data=data, columns=COLUMNS)
    # scaled_data.to_csv(f"scaled_{STATE}_data.csv")

    print("Scaled data:")
    print(scaled_data.head())
    print()

    # Unscale data
    unsacled_data = transformer.unscale_data(
        data=scaled_data, columns=COLUMNS, fitted_scaler=fitted_scaler
    )

    # unsacled_data.to_csv(f"unscaled_{STATE}_data.csv")
    print("Unscaled data:")
    print(unsacled_data.head())
    print()


if __name__ == "__main__":
    main()
