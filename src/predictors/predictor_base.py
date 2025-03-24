# Standard library imports
import pandas as pd
import logging
from typing import Union

# Custom imports
from src.utils.log import setup_logging
from src.local_model.model import BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.global_model.model import GlobalModel
from src.local_model.statistical_models import LocalARIMA

# Get pre-configured logger
logger = logging.getLogger("demograpy_predictor")


class DemographyPredictor:

    def __init__(
        self,
        name: str,
        local_model: Union[BaseLSTM, LocalARIMA],
        global_model: Union[GlobalModel],
    ):
        # Define name of the model -> name is used to versionning etc
        self.name: str = name

        # Define architecture: local model -> global model
        self.local_model: Union[BaseLSTM, LocalARIMA, FineTunableLSTM] = local_model
        self.global_model: Union[GlobalModel] = global_model

    def __repr__(self) -> str:
        return f"{self.name} ({type(self.local_model).__name__} -> {type(self.global_model.model).__name__})"

    def predict(self, input_data: pd.DataFrame, target_year: int) -> pd.DataFrame:

        # Preprocess input data -> scale data
        data = input_data.copy()

        # Features
        LOCAL_FEAUTRES = self.local_model.FEATURES

        # Scale local model data
        scaled_data = self.local_model.scaler.transform(data[LOCAL_FEAUTRES])
        scaled_data_df = pd.DataFrame(scaled_data, columns=LOCAL_FEAUTRES)

        # Get last year of predictions
        last_year = input_data["year"].max()

        # Predict features
        feature_predictions = self.local_model.predict(
            input_data=scaled_data_df, last_year=last_year, target_year=target_year
        )

        feature_predictions_df = pd.DataFrame(
            feature_predictions, columns=LOCAL_FEAUTRES
        )

        # Unscale feature predictions
        feature_predictions_unscaled = self.local_model.scaler.inverse_transform(
            feature_predictions
        )

        feature_predictions_df = pd.DataFrame(
            feature_predictions_unscaled, columns=LOCAL_FEAUTRES
        )

        # Add missing features
        GLOBAL_FEATURES = self.global_model.FEATURES

        adjust_for_gm_features_df = feature_predictions_df.copy()
        # Add country name
        adjust_for_gm_features_df["country name"] = data["country name"][0]
        adjust_for_gm_features_df["country name"] = adjust_for_gm_features_df[
            "country name"
        ].astype("category")

        # Add year
        adjust_for_gm_features_df["year"] = pd.Series(
            list(range(last_year, target_year + 1))
        )

        if set(adjust_for_gm_features_df.columns) != set(GLOBAL_FEATURES):
            raise ValueError(
                f"Missing features for global model: {set(adjust_for_gm_features_df.columns) ^ set(GLOBAL_FEATURES)}"
            )

        # # Scale using global scaler
        # scaled_feature_predictions = self.global_model.transform_columns(
        #     data=adjust_for_gm_features_df, columns=GLOBAL_FEATURES
        # )

        # scaled_feature_predictions_df = pd.DataFrame(
        #     scaled_feature_predictions, columns=feature_predictions_df.columns
        # )

        preprocessed_for_gm = self.global_model.preprocess_data(
            data=adjust_for_gm_features_df
        )

        # Predict target variable using global model
        final_predictions = self.global_model.predict_human_readable(
            data=preprocessed_for_gm[GLOBAL_FEATURES]
        )

        return final_predictions
