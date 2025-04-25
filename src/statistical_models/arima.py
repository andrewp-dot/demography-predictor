# Standard library imports
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import List, Optional

from statsmodels.tsa.arima.model import ARIMA

# Get logger
logger = logging.getLogger("local_model")

# Set the numpy random seed for reproducibility
np.random.seed(42)


class CustomARIMA:

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
        self.model: Optional[ARIMA] = None

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
            endog=target_values,
            exog=exog_values,
            order=(self.p, self.d, self.q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        # Fit the model
        self.model = new_model.fit(method_kwargs={"maxiter": 1000})
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
