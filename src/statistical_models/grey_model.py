# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# Note: This script is not finished yet!


class CustomGM11:

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

        raise NotImplementedError(
            f"The '{self.__class__.__name__}' is not fully implemented yet!"
        )

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
        rmse = root_mean_squared_error(TEST_DATA_NUMPY, predictions)
        r2 = r2_score(TEST_DATA_NUMPY, predictions)

        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R^2: {r2}")
