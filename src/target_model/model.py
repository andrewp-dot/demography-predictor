# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import pandas as pd
import logging
from typing import Union, List, Dict, Tuple, Optional
from pydantic import BaseModel


from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


from sklearn.model_selection import TimeSeriesSplit

## Import models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# Custom imports
from config import Config
from src.utils.log import setup_logging

from src.utils.constants import (
    basic_features,
    highly_correlated_features,
    aging_targets,
)

from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

# TODO: implement XGBoost using data

# TODO: implement XGBoost for predicing:
# 1. 'population, total' or 'population growth'
# 2. age distribution
# 3. gender distribution..?  (You can use maybe smaller dataset ...)

# Note for multioutput traininig:
# Final Recommendation
# For small datasets with 2-3 targets, train separate models.
# For large datasets with many outputs, use MultiOutputRegressor.

# TODO: fix data preparation
# 1. transform any input data to model input data

settings = Config()

logger = logging.getLogger("global_model")


class XGBoostTuneParams(BaseModel):
    n_estimators: List[int]  # Number of boosting rounds
    learning_rate: List[float]  # Step size shrinkage
    max_depth: List[int]  # Depth of trees
    subsample: List[float]  # Fraction of samples used per tree
    colsample_bytree: List[float]  # Fraction of features used per tree

    def to_regressor_dict(self, is_multitarget: bool) -> Dict[str, List[int | float]]:

        # Get the original dict
        original_dict = self.model_dump()

        # Get the dict of the original
        if not is_multitarget:
            return original_dict

        # Modify keys with 'estimator__' prefix
        modified_dict: Dict[str, List[int | float]] = {}
        for key, value in original_dict.items():
            modified_dict[f"estimator__{key}"] = value

        return modified_dict


class TargetModelTree:

    def __init__(
        self,
        model: Union[XGBRegressor, RandomForestRegressor, LGBMRegressor, GridSearchCV],
        features: List[str],
        targets: List[str],
        sequence_len: int,
        params: Optional[XGBoostTuneParams] = None,
        to_compute_target: Optional[str] = None,
    ):
        # Define model
        self.model: Union[
            XGBRegressor, RandomForestRegressor, LGBMRegressor, MultiOutputRegressor
        ] = model

        self.params: XGBoostTuneParams = params

        # Define features and targets

        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

        # Get the pridiction targets
        self.PREDICTION_TARGETS: List[str] = targets

        if to_compute_target:
            self.PREDICTION_TARGETS = [
                target for target in targets if target != to_compute_target
            ]

        # Save the to copmute target
        self.to_compute_target = to_compute_target

        self.sequence_len: int = sequence_len
        self.HISTORY_TARGET_FEATURES: List[str] = [
            f"{col}_t-{i}"
            for col in self.PREDICTION_TARGETS
            for i in range(self.sequence_len - 1, -1, -1)
        ]

        # Initialize evaluation results
        self.evaluation_results: pd.DataFrame | None = None

    def get_target_history(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates from the past target values creates features for as an input features for the specific timestep.

        Args:
            input_data (pd.DataFrame): All input data including past data

        Returns:
            out: pd.DataFrame: Dataframe with the faltten targets previous values.
        """

        # Add lagged features for the target data
        last_n_targets = input_data[self.PREDICTION_TARGETS].tail(self.sequence_len)

        # Switch rows and columns and create 1D array -> unpacking the history of targets
        flattened = last_n_targets.to_numpy().T.flatten()

        return pd.DataFrame([flattened], columns=self.HISTORY_TARGET_FEATURES)

    def create_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Get the last sequence_len data
        if len(input_data) < self.sequence_len:
            raise ValueError(
                f"Insufficient input data. Need at least {self.sequence_len} last records of input values."
            )

        last_n_data: pd.DataFrame = input_data.tail(self.sequence_len)

        # From the input sequence get the targets from time (T - (sequence_length + 1)) to time T
        target_history_df = self.get_target_history(input_data=last_n_data)

        xgb_input = pd.concat(
            [
                # From the input sequence get features in time T
                last_n_data.tail(1).reset_index(drop=True),
                target_history_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return xgb_input[self.FEATURES + self.HISTORY_TARGET_FEATURES]

    def create_state_inputs_outputs(
        self, states_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:

        # Create input and output sequences
        state_inputs_dict: Dict[str, List[pd.DataFrame]] = {}
        state_outputs_dict: Dict[str, List[pd.DataFrame]] = {}
        for state, df in states_dict.items():

            # Create rolling window
            number_of_samples = df.shape[0] - self.sequence_len

            # Skip if not enough data
            if number_of_samples <= 0:
                continue

            inputs = []
            outputs = []

            # Pre-extract values for faster slicing
            df_values = df.values
            df_columns = df.columns

            target_indices = [df_columns.get_loc(target) for target in self.TARGETS]

            for i in range(number_of_samples):

                # Add input sequence
                window = df.iloc[i : i + self.sequence_len]

                xgb_input = self.create_input(input_data=window)

                inputs.append(xgb_input)

                # Add sequence output - and convert series to dataframe
                # Extract output (targets after window)
                next_target_row = df_values[i + self.sequence_len, target_indices]
                next_target_df = pd.DataFrame([next_target_row], columns=self.TARGETS)
                outputs.append(next_target_df)

            # Stack inputs and outputs once (much faster than incremental appends)
            state_inputs_dict[state] = pd.concat(inputs, ignore_index=True)
            state_outputs_dict[state] = pd.concat(outputs, ignore_index=True)

        return state_inputs_dict, state_outputs_dict

    def create_train_test_timeseries(
        self,
        states_dfs: Dict[str, pd.DataFrame],
        states_loader: StatesDataLoader,
        split_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        From the given dataframe creates and scales the data using fitted scaler. Splits the dataframes by time: train data are first `len(X) * split_size` occurences, test data are the rest.

        Args:
            states_dfs (Dict[str, pd.DataFrame]): Dictionary of state dataframes where state name is the key.
            states_loader (StatesDataLoader): Class used to split the data by timeseries.
            split_size (float): The size of the train part.
            fitted_scaler (MinMaxScaler): Scaler used to scale the other features from the previous prediction model.

        Returns:
            out: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
        """

        states_inputs_dict, states_outputs_dict = self.create_state_inputs_outputs(
            states_dict=states_dfs
        )

        # Split inputs
        X_train_dfs, X_test_dfs = states_loader.split_data(
            states_dict=states_inputs_dict,
            sequence_len=1,
            split_rate=split_size,
        )

        # Split outputs
        y_train_dfs, y_test_dfs = states_loader.split_data(
            states_dict=states_outputs_dict,
            sequence_len=1,
            split_rate=split_size,
        )

        # Convert inputs to dataframe
        X_train = states_loader.merge_states(state_dfs=X_train_dfs)
        y_train = states_loader.merge_states(state_dfs=y_train_dfs)

        X_test = states_loader.merge_states(state_dfs=X_test_dfs)
        y_test = states_loader.merge_states(state_dfs=y_test_dfs)

        # Set the country name as the categorical column
        if "country_name" in X_train.columns:
            X_train["country_name"] = X_train["country_name"].astype(dtype="category")

        if "counrty name" in X_test.columns:
            X_test["country_name"] = X_test["country_name"].astype(dtype="category")

        # Create X
        # X_train, y_train = train_df[self.FEATURES], train_df[self.TARGETS]
        # X_test, y_test = test_df[self.FEATURES], test_df[self.TARGETS]

        return (
            X_train[self.FEATURES + self.HISTORY_TARGET_FEATURES],
            X_test[self.FEATURES + self.HISTORY_TARGET_FEATURES],
            y_train,
            y_test,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        tune_hyperparams: bool = False,
    ) -> None:

        # Find out whether training multioutput regressor
        IS_MULTI_TARGET: bool = len(self.TARGETS) > 1

        # Create multioutput regressor for for the targets
        param_dict = None

        if self.params:
            param_dict = self.params.to_regressor_dict(is_multitarget=IS_MULTI_TARGET)

        # Create multioutput regressor for for the targets
        if IS_MULTI_TARGET:
            self.model = MultiOutputRegressor(self.model)

        # Tune hyperparams if neaded
        if tune_hyperparams:
            tscv = TimeSeriesSplit(n_splits=3)

            if param_dict is not None:
                logger.info("Tuning parameters....")
                self.model = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_dict,
                    cv=tscv,
                    scoring="neg_mean_squared_error",  # Minimize MSE
                    verbose=1,
                    n_jobs=2,  # Use all available CPUs
                )
            else:
                logger.warning(
                    f"Tune hyperparameters is set to True, but no hyperparameters were provided. Skipping parameter tuning."
                )

        # Fit the model
        logger.info("Fitting model...")

        self.model.fit(X_train, y_train)
        logger.info("Model succesfuly fitted!")

    def eval(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """
        Evaluates model using test data. Saves the data into the `evaluation_results` parameter of the TargetModelTree class.

        Args:
            X_test (pd.DataFrame): Test input data.
            y_test (pd.DataFrame): Test output data.
        """

        # Get predictions in human readable dataframe
        predictions_df = self.predict_human_readable(X_test)

        # Compute evaluation metrics
        mse = mean_squared_error(y_test, predictions_df)
        rmse = root_mean_squared_error(y_test, predictions_df)  # RMSE
        mae = mean_absolute_error(y_test, predictions_df)
        r2 = r2_score(y_test, predictions_df)

        # Optionally, store metrics in an attribute for later use
        self.evaluation_results = pd.DataFrame(
            {"mse": [mse], "rmse": [rmse], "mae": [mae], "r2": [r2]}
        )

        logger.info(f"TargetModelTree evaluation:\n{self.evaluation_results}")

    def predict_human_readable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts and unscales the scaled data.

        Returns:
            out: pd.DataFrame: The pandas dataframe with human readable predictions to next year.
        """

        # Adjust input to the correct format if needed
        if not all(feature in data.columns for feature in self.HISTORY_TARGET_FEATURES):
            xgb_input = self.create_input(data)

        else:
            xgb_input = data[self.FEATURES + self.HISTORY_TARGET_FEATURES]

        predictions = self.model.predict(xgb_input)

        # Create predictions dataframe
        predictions_df = pd.DataFrame(predictions, columns=self.PREDICTION_TARGETS)

        return predictions_df


def try_single_target_global_model():

    # Load data
    logger.info("Loading whole dataset....")
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    # Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    targets: List[str] = aging_targets()
    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    # Tune params
    # Note: too many parameters results in a warning
    tune_parameters = XGBoostTuneParams(
        n_estimators=[50, 100],
        learning_rate=[0.001, 0.01, 0.05],
        max_depth=[3, 5, 7],
        subsample=[0.5, 0.7],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    # Create global model
    gm = TargetModelTree(
        model=XGBRegressor(objective="reg:squarederror", random_state=42),
        features=FEATURES,
        targets=targets,
        params=tune_parameters,
        sequence_len=5,
    )

    logger.info("Training the model....")

    # Create train and test data
    X_train, X_test, y_train, y_test = gm.create_train_test_timeseries(
        states_dfs=state_dfs,
        states_loader=states_loader,
        split_size=0.8,
    )

    # Train model
    gm.train(
        X_train=X_train,
        y_train=y_train,
        # tune_hyperparams=True,
    )

    # Evaluate model
    logger.info("Evaluating the model...")
    gm.eval(X_test=X_test, y_test=y_test)

    pred = gm.predict_human_readable(data=state_dfs["Czechia"])
    print(pred)


def try_sequences_creation():

    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()
    # state_dfs = states_loader.load_states(states=["Czechia", "United States"])

    # Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    # Targets
    # targets: List[str] = ["population, total"]
    targets: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]

    # # Features
    FEATURES = [
        col.lower()
        for col in [
            # "year",
            "Fertility rate, total",
            "Population, total",
            "Net migration",
            "Arable land",
            # "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            # "Rural population",
            "Rural population growth",
            # "Age dependency ratio",
            "Urban population",
            "Population growth",
            # "Adolescent fertility rate",
            # "Life expectancy at birth, total",
            # Population total target
        ]
    ]

    gm = TargetModelTree(
        model=XGBRegressor(), features=FEATURES, targets=targets, sequence_len=10
    )
    inputs, outputs = gm.create_state_inputs_outputs(states_dict=state_dfs)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Try global model
    try_single_target_global_model()
