# Standard library imports
import logging
import pandas as pd
import numpy as np
from typing import List

from statsmodels.tsa.arima.model import ARIMA

# Custom imports
from config import Config, setup_logging
from src.preprocessors.state_preprocessing import StateDataLoader

# Get logger
logger = logging.getLogger("local_model")

# Set the numpy random seed for reproducibility
np.random.seed(42)


class LocalARIMAHyperparams:

    def __init__(self):
        pass


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
        train_data.set_index(self.index, inplace=True, drop=True)

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

    def predict(self, data: pd.DataFrame, steps: int) -> pd.Series:

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
        # Choose the appropriate method for prediction
        if steps is None:
            predictions = self.model.predict(exog=exog_values)  # In-sample prediction
        else:
            predictions = self.model.predict(
                start=len(data), end=len(data) + steps - 1, exog=exog_values
            )

        return predictions

    def eval(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Evaluates the model predictions based on train and test data.
        """

        # Get copies of the data
        train_data = train_df.copy()
        test_data = test_df.copy()

        # Set index of the dataframes
        train_data.set_index(self.index, inplace=True)
        test_data.set_index(self.index, inplace=True)

        # Setup predictions

        # Predict
        predictions = self.predict(data=train_data, steps=6)

        # Compare predictions

        # TODO: add metrics


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
    FEATURES: List[str] = []
    TARGET: str = "population, total"

    model = LocalARIMA(1, 1, 1, features=FEATURES, target=TARGET, index="year")

    # Train model
    model.train_model(state_df)

    # Evaluate model
    model.eval(train_df=train_df, test_df=test_df)

    # Get and print the evalutaion


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    try_arima(state="Czechia", split_rate=0.8)
