# Standard library imports
import pandas as pd
from typing import Union, List
from pydantic import BaseModel

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


## Import models
from xgboost import XGBRegressor

# Custom imports

# TODO: implement XGBoost using data


class XGBoostTuneParams(BaseModel):
    n_estimators: List[int]  # Number of boosting rounds
    learning_rate: List[float]  # Step size shrinkage
    max_depth: List[int]  # Depth of trees
    subsample: List[float]  # Fraction of samples used per tree
    colsample_bytree: List[float]  # Fraction of features used per tree


class GlobalModel:

    def __init__(
        self,
        model: Union[XGBRegressor, GridSearchCV],
        features: List[str],
        targets: List[str],
        params: XGBoostTuneParams | None = None,
    ):

        # Define model
        self.model: Union[XGBRegressor] = model
        self.params: XGBoostTuneParams = params

        # Define features and targest
        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # First scale, sort, normalize data for input
        raise NotImplementedError("")

    def train_and_eval(
        self, data: pd.DataFrame, split_size: float, tune_hyperparams: bool = False
    ) -> None:

        # Preprocess data
        preprocessed_data = self.preprocess_data(data=data)

        # Split data
        X: pd.DataFrame = preprocessed_data[self.FEATURES]
        y: pd.DataFrame = preprocessed_data[self.TARGETS]

        X_train, y_train, X_test, y_test = train_test_split(
            X, y, test_size=(1 - split_size)
        )

        # Tune hyperparams if neaded
        if tune_hyperparams:
            self.model = GridSearchCV(
                estimator=self.model,
                param_grid=self.params,
                scoring="neg_mean_squared_error",  # Minimize MSE
                cv=3,
                verbose=1,
                n_jobs=-1,  # Use all available CPUs
            )

        # Fit the model
        self.model.fit(X_train, y_train)

        # Training stats

        raise NotImplementedError("")

    def eval(self, test_df: pd.DataFrame) -> None:
        NotImplementedError("")
