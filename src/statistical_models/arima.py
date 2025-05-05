# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
from itertools import product

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tools.eval_measures import aic
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# Get logger
logger = logging.getLogger("local_model")

# Set the numpy random seed for reproducibility
np.random.seed(42)

# ARIMA model optimization
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/


class CustomARIMA:

    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        features: List[str],
        target: str,
        index: str,
        trend: Optional[str] = None,
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
            trend (Optional[str]): Parameter which controls whether the model automatically includes a constant term (or a linear trend). Defaults to None.
        """

        # Parameters of ARIMA
        self.p: int = p
        self.d: int = d
        self.q: int = q

        # Input and target variable(s)
        self.features: List[str] = features
        self.target: str = target
        self.index: str = index
        self.trend: Optional[str] = trend

        # Create the model
        self.model: Optional[ARIMA] = None

    def __repr__(self) -> str:
        return f"ARIMA({self.p}, {self.d}, {self.q})"

    def find_best_params(
        self,
        target_values: np.ndarray,
        exog_values: np.ndarray,
        max_p: int = 3,
        max_d: int = 3,
        max_q: int = 3,
    ) -> Tuple[int, int, int]:
        best_aic = np.inf
        best_order = (0, 0, 0)

        # Iterate over possible p, d, q values
        for p, d, q in product(range(max_p), range(max_d), range(max_q)):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = ARIMA(
                        endog=target_values,
                        exog=exog_values,
                        order=(p, d, q),
                        enforce_stationarity=True,
                        enforce_invertibility=False,
                        trend=self.trend,
                    )

                    model_fit = model.fit()
                current_aic = model_fit.aic
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order = (p, d, q)
            except KeyboardInterrupt as e:
                raise KeyboardInterrupt(e)
            except:
                continue

        return best_order

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
            enforce_stationarity=True,
            enforce_invertibility=False,
            trend=self.trend,
        )

        # Try to fit predefined arima
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                self.model = new_model.fit(method_kwargs={"maxiter": 1000})
                logger.info(f"{self} model fitted!")
                return

        except ConvergenceWarning:
            logger.info(f"ConvergenceWarning: searching for better parameters...")

        p, d, q = self.find_best_params(
            target_values=target_values, exog_values=exog_values
        )

        # Set the params
        self.p = p
        self.d = d
        self.q = q

        new_model = ARIMA(
            endog=target_values,
            exog=exog_values,
            order=(self.p, self.d, self.q),
            enforce_stationarity=True,
            enforce_invertibility=False,
            trend=self.trend,
        )

        # Fit the model
        self.model = new_model.fit(method_kwargs={"maxiter": 1000})
        logger.info(f"{self} model fitted!")

    def predict(self, data: pd.DataFrame, steps: int) -> pd.DataFrame:
        # Ensure model is trained
        if self.model is None:
            raise ValueError(
                "The ARIMA model is not trained. Did you call train_model() first?"
            )

        data_copy = data.copy()

        # Find known target values (non-NaN)
        known_target = data_copy[self.target].dropna()
        start = len(known_target)
        end = start + steps - 1

        # To predict target values are set to None -> these rows also contains the known exogeneous variables

        # Extract corresponding future exogenous variables
        try:
            if self.features:
                future_rows = data_copy[self.target].isna()
                exog_values = data_copy.loc[future_rows, self.features]
                if len(exog_values) != steps:
                    raise ValueError(
                        f"Expected {steps} rows of exogenous features, got {len(exog_values)}"
                    )
            else:
                exog_values = None
        except KeyError as e:
            raise KeyError(f"The {e} column is not in the dataframe!")

        # Predict future values

        predictions = self.model.predict(start=start, end=end, exog=exog_values)

        # Create a DataFrame and align index to the future time steps
        prediction_df = predictions.to_frame(name=self.target)

        # If there is known features, use known index, otherwise keep the old index
        if self.features:
            prediction_df.index = data_copy.loc[future_rows].index

        # Rename the predicted mean to target name
        prediction_df.rename(columns={"predicted_mean": self.target}, inplace=True)
        return prediction_df.iloc[-steps:]
