# Standard library imports
import pandas as pd
import numpy as np
import logging
from typing import Union, List, Dict, Tuple
from pydantic import BaseModel
import shap

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


## Import models
import xgboost as xgb
from xgboost import XGBRegressor


# Custom imports
from config import Config
from src.utils.log import setup_logging

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


class GlobalModel:

    def __init__(
        self,
        model: Union[XGBRegressor, GridSearchCV],
        features: List[str],
        targets: List[str],
        params: XGBoostTuneParams | None = None,
    ):

        # Define model
        self.model: Union[XGBRegressor, MultiOutputRegressor] = model
        self.params: XGBoostTuneParams = params

        # Define features and targest
        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

        # Initialize evaluation results
        self.evaluation_results: pd.DataFrame | None = None

    def create_train_test_data(
        self,
        data: pd.DataFrame,
        split_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        From the given dataframe creates and scales the data using fitted scaler. Use train_test_split function from sklearn.

        Args:
            data (pd.DataFrame): Preprocessed data for training split.
            split_size (float): The size of the train part.
            fitted_scaler (MinMaxScaler): Scaler used to scale the other features from the previous prediction model.

        Returns:
            out: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
        """

        # Set the country name as the categorical column
        if "country name" in data.columns:
            data.loc[:, "country name"] = data["country name"].astype("category")

        # Split data
        X: pd.DataFrame = data[self.FEATURES]
        y: pd.DataFrame = data[self.TARGETS]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(1 - split_size),
            random_state=42,
        )

        return X_train, X_test, y_train, y_test

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

        # Split by time series, sequence len is equal to 1 in order to include all states
        states_train_dfs, states_test_dfs = states_loader.split_data(
            states_dict=states_dfs, sequence_len=1, split_rate=split_size
        )

        # Create X and y
        train_df: pd.DataFrame = states_loader.merge_states(state_dfs=states_train_dfs)
        test_df: pd.DataFrame = states_loader.merge_states(state_dfs=states_test_dfs)

        # Set the country name as the categorical column
        if "country name" in train_df.columns:
            train_df["country name"] = train_df["country name"].astype(dtype="category")

        if "counrty name" in test_df.columns:
            test_df["country name"] = test_df["country name"].astype(dtype="category")

        # Create X
        X_train, y_train = train_df[self.FEATURES], train_df[self.TARGETS]
        X_test, y_test = test_df[self.FEATURES], test_df[self.TARGETS]

        return X_train, X_test, y_train, y_test

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

            if param_dict is not None:
                logger.info("Tuning parameters....")
                self.model = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_dict,
                    scoring="neg_mean_squared_error",  # Minimize MSE
                    cv=3,
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
        Evaluates model using test data. Saves the data into the `evaluation_results` parameter of the GlobalModel class.

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

        logger.info(f"GlobalModel evaluation:\n{self.evaluation_results}")

    def predict_human_readable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts and unscales the scaled data.

        Returns:
            out: pd.DataFrame: The pandas dataframe with human readable predictions.
        """
        predictions = self.model.predict(data)

        # Create predictions dataframe
        predictions_df = pd.DataFrame(predictions, columns=self.TARGETS)

        return predictions_df

    # def feature_importance(self, X_train: pd.DataFrame):
    #     """
    #     Create a plot or something using SHAP explainer.
    #     """

    #     if None is None:
    #         raise NotImplementedError(
    #             "Need to implement this for explaining features! Maybe you can implement training and for single prediction feature importance explainer."
    #         )

    #     explainer: shap.Explainer = shap.Explainer(self.model)
    #     shap_values = explainer(X_train)

    #     shap.summary_plot(shap_values, X_train)

    #     # For a single prediction
    #     shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[0])

    #     # Shows interaction effects
    #     shap.dependence_plot("feature_name", shap_values, X_train)


def try_single_target_global_model():

    # Load data
    logger.info("Loading whole dataset....")
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    # Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    # Targets
    # targets: List[str] = ["population, total"]
    targets: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]

    # Features
    features: List[str] = [
        col for col in whole_dataset_df.columns if col not in targets
    ]

    # Tune params
    # Note: too many parameters results in a warning
    tune_parameters = XGBoostTuneParams(
        n_estimators=[50, 100],
        learning_rate=[0.001, 0.01, 0.05],
        max_depth=[3, 5, 7],
        subsample=[0.5, 0.7],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    # Simulation of scaler used to scale the data from the local model
    scaler = MinMaxScaler()
    fitted_scaler = scaler.fit(whole_dataset_df.drop(columns=["country name"]))

    # Create global model
    gm = GlobalModel(
        model=XGBRegressor(objective="reg:squarederror", random_state=42),
        features=features,
        targets=targets,
        params=tune_parameters,
        scaler=fitted_scaler,
    )

    logger.info("Training the model....")

    # Create train and test data
    X_train, X_test, y_train, y_test = gm.create_train_test_data(
        data=whole_dataset_df,
        split_size=0.8,
        fitted_scaler=fitted_scaler,
    )

    # Creater train and test data with timeseries
    # X_train, X_test, y_train, y_test = gm.create_train_test_timeseries(
    #     states_dfs=state_dfs,
    #     states_loader=states_loader,
    #     split_size=0.8,
    #     fitted_scaler=fitted_scaler,
    # )

    # Train model
    gm.train(
        X_train=X_train,
        y_train=y_train,
        # tune_hyperparams=True,
    )

    # Evaluate model
    logger.info("Evaluating the model...")
    gm.eval(X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Try global model
    try_single_target_global_model()
