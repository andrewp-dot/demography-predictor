# Standard library imports
import pandas as pd
import logging
from typing import Union, List
from pydantic import BaseModel

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


## Import models
import xgboost as xgb
from xgboost import XGBRegressor


# Custom imports
from config import Config
from src.utils.log import setup_logging

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
        self,
        data: pd.DataFrame,
        split_size: float,
        fitted_scaler: MinMaxScaler,
        tune_hyperparams: bool = False,
    ) -> None:

        # Set the country name as the categorical column
        if "country name" in data.columns:
            data["country name"] = data["country name"].astype(dtype="category")

        # Preprocess data
        logger.info("Preprocessing data....")
        preprocessed_data = self.preprocess_data(data=data, fitted_scaler=fitted_scaler)

        logger.critical(preprocessed_data.columns)

        # Split data
        X: pd.DataFrame = preprocessed_data[self.FEATURES]
        y: pd.DataFrame = preprocessed_data[self.TARGETS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - split_size), random_state=42
        )

        # Convert dataset into DMatrix format
        # if len(self.TARGETS) > 1:
        #     raise NotImplementedError("Multitarget training not implemented yet!")

        # Create train and test DMatricies
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Tune hyperparams if neaded
        if tune_hyperparams and self.params is not None:
            logger.info("Tuning parameters....")
            self.model = GridSearchCV(
                estimator=self.model,
                param_grid=self.params,
                scoring="neg_mean_squared_error",  # Minimize MSE
                cv=3,
                verbose=1,
                n_jobs=-1,  # Use all available CPUs
            )

        # Fit the model
        logger.info("Fitting model...")

        params = {"objective": "reg:squarederror", "eval_metric": "rmse"}

        self.model = xgb.train(
            params, dtrain, num_boost_round=100, evals=[(dtest, "Test")]
        )

    def eval(self, test_df: pd.DataFrame) -> None:
        NotImplementedError("")


def try_single_target_global_model():

    # Load data
    whole_dataset_df = pd.read_csv(settings.dataset_path)
    logger.info("Loading whole dataset....")

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
    tune_parameters = XGBoostTuneParams(
        n_estimators=[50, 100, 200],
        learning_rate=[0.001, 0.01, 0.05, 0.1],
        max_depth=[3, 5, 7, 10],
        subsample=[0.5, 0.7, 0.9, 1.0],
        colsample_bytree=[0.5, 0.7, 0.9, 1.0],
    )

    # Create global model
    gm = GlobalModel(
        model=XGBRegressor(),
        features=features,
        targets=targets,
        params=tune_parameters,
    )

    logger.info("Training and evaluation model....")

    # Simulation of scaler used to scale the data from the local model
    scaler = MinMaxScaler()
    fitted_scaler = scaler.fit(whole_dataset_df.drop(columns=["country name"]))

    gm.train_and_eval(
        data=whole_dataset_df,
        split_size=0.8,
        fitted_scaler=fitted_scaler,
        tune_hyperparams=True,
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Try global model
    try_single_target_global_model()
