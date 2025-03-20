# Standard library imports

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Callable
from matplotlib.figure import Figure

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.preprocessors.state_preprocessing import StateDataLoader
from src.local_model.base import BaseEvaluation

# Get logger
logger = logging.getLogger("local_model")

# Set the numpy random seed for reproducibility
np.random.seed(42)


class EvaluateARIMA(BaseEvaluation):

    def __init__(self, arima: LocalARIMA):

        super().__init__()

        # Get evaluation data
        self.model: LocalARIMA = arima

    def __get_metric(
        self,
        metric: Callable,
        metric_key: str = "",
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Computes and saves given metric.

        Args:
            metric (Callable): Function for metric computation.
            features (List[str]): Key of the metric which can be accasible in `EvaluateModel.metrics` dict (`{metric_key}`, `{metric_key}_per_target` if per target is available).
                If not given, the name of the function is used. Defaults to "".
            metric_key (str, optional): _description_. Defaults to "".

        Returns:
            out: Tuple[pd.DataFrame, pd.DataFrame | None]: metric value for all targets, metric values for each target separately.
        """

        # Adjust metric key if not given
        metric_key = metric_key or metric.__name__

        # Initialize metric values
        overall_metric_df = None

        # Average MAE across all targets
        average_metric_value = metric(
            self.reference_values, self.predicted, multioutput="uniform_average"
        )

        # Get overall metric dataframe
        overall_metric_df = pd.DataFrame(
            {"metric": metric_key, "value": [average_metric_value]}
        )

        return overall_metric_df

    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        features: list[str],
        target: str,
        index: str,
    ):

        # Set features as a constant
        # FEATURES = features

        # Get the last year and get the number of years
        X_years = test_X[["year"]]
        last_year = int(X_years.iloc[-1].item())

        # # Get the prediction year
        y_target_years = test_y[["year"]]
        target_year = int(y_target_years.iloc[-1].item())

        # Calculate steps
        steps = target_year - last_year

        # Get predicted years
        self.predicted_years = range(last_year + 1, target_year + 1)

        # Get copies of the data
        train_data = test_X.copy()
        test_data = test_y.copy()

        # Set index of the dataframes
        train_data.set_index(index, inplace=True)
        test_data.set_index(index, inplace=True)

        # Save true values
        self.reference_values = test_data[target].to_frame()

        # Get predictions
        self.predicted = self.model.predict(data=test_data, steps=steps)

        # Get overall metrtics
        self.get_overall_metrics()


class LocalARIMA:

    def __init__(
        self, p: int, d: int, q: int, features: List[str], target: str, index: str
    ):
        """
        Create custom ARIMA model: ARIMA(p,d,q).

        Args:
            p (int): Number of lagged (previous) observations (AR component).
            d (int): Number of differences needed to make the series stationary. If the `d = 0` it is just ARMA model.
            q (int): Number of lagged forecast errors (MA component). The lagged errors are used to correct next predictions.
            features (List[str]): List of features columns
            target (str): Values to predict.
            index (str): Features used for index (usually 'year', or 'date' - time columns)
        """

        # Parameters of ARIMA
        self.p: int = p
        self.d: int = d
        self.q: int = q

        # Input and target variable(s)
        self.features: List[str] = features
        self.target: str = target
        self.index: str = index

        # Create the model
        self.model: ARIMA | None = None

    def __repr__(self) -> str:
        return f"ARIMA({self.p}, {self.d}, {self.q})"

    # Do this just for one variable or for multiple variable?
    def train_model(self, data: pd.DataFrame) -> None:

        # Set index of dataframe
        train_data = data.copy()
        train_data.set_index(self.index, inplace=True)

        # Get the values of the features
        exog_values = None
        if self.features:
            exog_values = data[self.features]

        # Get target values
        target_values = data[self.target]

        # Create the arima model
        new_model = ARIMA(
            endog=target_values, exog=exog_values, order=(self.p, self.d, self.q)
        )

        # Fit the model
        self.model = new_model.fit()
        logger.info(f"ARIMA model fitted!")

    def predict(self, data: pd.DataFrame, steps: int) -> pd.DataFrame:

        # Try if the model is trained
        if self.model is None:
            raise ValueError(
                "The ARIMA model is not trained. Did you call train_model() first?"
            )

        # Extract the prediction data
        try:
            if self.features:
                exog_values = data[self.features]
            else:
                exog_values = None
        except KeyError as e:
            raise KeyError(f"The {e} column is not in the dataframe!")

        # Predict
        predictions = self.model.predict(
            start=len(data), end=len(data) + steps - 1, exog=exog_values
        )

        # Convert prediction series to dataframe
        predition_df: pd.DataFrame = predictions.to_frame()

        # Rename the predicted mean to target name
        predition_df.rename(columns={"predicted_mean": self.target}, inplace=True)
        return predition_df

    def eval(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Figure:
        """
        Evaluates the model predictions based on train and test data.
        """

        # Setup predictions
        arima_evaluation = EvaluateARIMA(self)
        arima_evaluation.eval(
            test_X=train_df,
            test_y=test_df,
            features=self.features,
            target=self.target,
            index=self.index,
        )

        # Compare predictions

        fig = arima_evaluation.plot_predictions()


# TODO: implement GM(1,1)


class LocalGM11:

    # 3 phases:
    # 1. AGO (Accumulated Generating Operation)
    # 2. GM(1,1) model - grey modeling
    # 3. IAGO (Inverse Accumulated Generating Operation)

    def __init__(self):
        """
        Initialize the parameters for the GM(1,1) model.
        """

        # Initialize the model
        self.X1_0: np.ndarray = None  # Initial value
        self.a: float = None
        self.b: float = None
        self.n: int = None  # Number of observations

        # Train data values
        self.train_data: pd.DataFrame = None
        self.train_data_numpy: np.ndarray = None

        # Test data values
        self.test_data: pd.DataFrame = None
        self.test_data_numpy: np.ndarray = None

    def train_model(self, data: pd.DataFrame, target: str, split_rate: float = 0.8):
        """
        Trains the GM(1,1) model using the given data.

        Note: GM(1,1) is single dimension grey model, which is used to predict the future values of a sequence.

        Args:
            data (pd.DataFrame): Given data to train GM(1,1) model.
            target (str): The target column to predict and fit.
            split_rate (float, optional): Split data to train and test data. Defaults to 0.8.
        """

        # Copy the given data
        data_copy = data.copy()

        # Get the data, convert them to numpy array and flatten them
        SPLIT_INDEX = int(len(data_copy) * split_rate)
        self.train_data = data_copy[target].iloc[:SPLIT_INDEX]
        self.test_data = data_copy[target].iloc[SPLIT_INDEX:]

        # Get the train data as numpy array
        self.train_data_numpy = data.to_numpy().flatten()
        self.n = len(self.train_data_numpy)

        # 1. Accumulate data (AGO)
        X1 = np.cumsum(self.train_data)

        # 2. Form B and Y matrices
        B = np.column_stack((-X1[:-1], np.ones(self.n - 1)))
        Y = self.train_data[1:].reshape(-1, 1)

        # 3. Solve for parameters (a, b)
        [[a], [b]] = np.linalg.lstsq(B, Y, rcond=None)[0]

        self.a, self.b = a, b
        self.X1_0 = self.data[0]

    def predict(self, m: int) -> np.ndarray:
        """
        Generate m predictions using the GM(1,1) model.

        Args:
            m (int): Number of predictions to generate.

        Returns:
            np.ndarray: Predicted values.
        """
        # Generate predicted sequence
        X1_pred = (self.X1_0 - self.b / self.a) * np.exp(
            -self.a * np.arange(self.n + m)
        ) + self.b / self.a
        X0_pred = np.diff(
            X1_pred, prepend=self.X1_0
        )  # Convert back to original sequence
        return X0_pred[-m:]  # Return the last m predictions

    def eval(self):
        """
        Evaluates the GM(1,1) model predictions based on train and test data.
        """

        # Convert the test data to numpy array if it was not converted yet
        if self.test_data_numpy is None:
            self.test_data_numpy = self.test_data.to_numpy().flatten()

        # Set the test data constant
        TEST_DATA_NUMPY = self.test_data_numpy

        # Predict the test data
        predictions = self.predict(len(TEST_DATA_NUMPY))

        # Compare predictions
        # TODO: Implement evaluation model
        mae = mean_absolute_error(TEST_DATA_NUMPY, predictions)
        mse = mean_squared_error(TEST_DATA_NUMPY, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(TEST_DATA_NUMPY, predictions)


def try_arima(state: str, split_rate: float):

    # Print statsmodels version
    import statsmodels

    logger.debug(f"Stats models version: {statsmodels.__version__}")

    # Load data
    STATE = state
    state_loader = StateDataLoader(STATE)

    state_df = state_loader.load_data()

    # Split data
    train_df, test_df = state_loader.split_data(state_df, split_rate=split_rate)

    # Create ARIMA
    FEATURES: List[str] = [
        # "year",
        # "Fertility rate, total",
        # "Population, total",
        # "Net migration",
        # "Arable land",
        # "Birth rate, crude",
        # "GDP growth",
        # "Death rate, crude",
        # "Agricultural land",
        # "Rural population",
        # "Rural population growth",
        # "Age dependency ratio",
        # "Urban population",
        # "Population growth",
        # "Adolescent fertility rate",
        # "Life expectancy at birth, total",
    ]
    TARGET: str = "arable land"

    model = LocalARIMA(1, 0, 1, features=FEATURES, target=TARGET, index="year")

    # Train model
    model.train_model(state_df)

    # Evaluate model
    model.eval(train_df=train_df, test_df=test_df)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    try_arima(state="Czechia", split_rate=0.8)
