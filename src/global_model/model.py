# Standard library imports
import pandas as pd
from typing import Union, List
from pydantic import BaseModel

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


## Import models
import xgboost as xgb
from xgboost import XGBRegressor


# Custom imports

# TODO: implement XGBoost using data

# TODO: implement XGBoost for predicing:
# 1. 'population, total' or 'population growth'
# 2. age distribution
# 3. gender distribution..?  (You can use maybe smaller dataset ...)

# Note for multioutput traininig:
# Final Recommendation
# For small datasets with 2-3 targets, train separate models.
# For large datasets with many outputs, use MultiOutputRegressor.


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

    def preprocess_data(
        self, data: pd.DataFrame, fitted_scaler: MinMaxScaler
    ) -> pd.DataFrame:

        # Copy dataframe
        df = data.copy()

        # Scale the numerical columns
        numerical_data_df = df.select_dtypes(include=["number"])

        # Scale numerical data
        scaled_numerical_data_array = fitted_scaler.transform(numerical_data_df)

        scaled_numerical_data_df = pd.DataFrame(
            scaled_numerical_data_array, columns=numerical_data_df.columns
        )

        categorical_data_df = df.select_dtypes(exclude=["number"])

        scaled_df = pd.concat([categorical_data_df, scaled_numerical_data_df], axis=1)

        return scaled_df

    def train_and_eval(
        self, data: pd.DataFrame, split_size: float, tune_hyperparams: bool = False
    ) -> None:

        # Preprocess data
        preprocessed_data = self.preprocess_data(data=data)

        # TODO: preprocess data to DMatrix
        # xgb.DMatrix(preprocessed_data, label=df["target"], enable_categorical=True)

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


if __name__ == "__main__":
    # import xgboost as xgb
    # import pandas as pd

    # Create a sample dataset
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "target": [10, 20, 30, 40, 50],
        }
    )

    # Convert dataset into DMatrix format
    dtrain = xgb.DMatrix(df.drop(columns=["target"]), label=df["target"])

    print(dtrain)
