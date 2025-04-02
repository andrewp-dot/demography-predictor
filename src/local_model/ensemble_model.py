# Standard library imports
import pandas as pd
from typing import Dict, Union, List

# Custom library imports
from src.local_model.base import CustomModelBase
from src.local_model.model import BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.statistical_models import LocalARIMA


class EnsembleModel(CustomModelBase):

    # TODO: implement this
    def __init__(self, features, targets, hyperparameters, scaler, *args, **kwargs):
        super().__init__(features, targets, hyperparameters, scaler, *args, **kwargs)

        # Hyperparameters will be used constantly for local models -> maybe I will use really class model

        # Model is a dictionary -> each feature has its own preidctor
        self.model: Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]] = {}

    def predict(self, input_data, last_year, target_year):
        return super().predict(input_data, last_year, target_year)

    def train_model(self, batch_inputs, batch_targets, display_nth_epoch=10):
        return super().train_model(batch_inputs, batch_targets, display_nth_epoch)


class PureEnsembleModel:

    def __init__(
        self, feature_models: Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]]
    ):
        # Get features
        self.FEATURES: List[str] = list(feature_models.keys())

        # Targets are the same
        self.TARGETS: List[str] = self.FEATURES

        self.model: Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]] = (
            feature_models
        )

    def predict(self, input_data: pd.DataFrame, target_year: int) -> pd.DataFrame:

        # Get last year
        if "year" not in input_data.columns:
            raise ValueError(
                "Missing column: 'year'. Cannot find out how many years to predict"
            )

        last_year = input_data.sort_values(by="year", ascending=True)[-1].item()

        # Preprocess data
        years_to_predict = range(last_year + 1, target_year + 1)

        # Init prediction df
        prediction_df: pd.DataFrame | None = None

        for year in years_to_predict:

            timestep_predictions: pd.DataFrame | None = None

            # Predict each feature for every year
            for feature in self.FEATURES:

                current_model = self.model[feature]
                # By type
                if isinstance(current_model, LocalARIMA):
                    feature_prediction_df = current_model.predict(
                        data=input_data, steps=years_to_predict
                    )

                if isinstance(current_model, BaseLSTM) or isinstance(
                    current_model, FineTunableLSTM
                ):
                    feature_prediction_df = current_model.predict(
                        input_data=input_data,
                        last_year=last_year,
                        target_year=target_year,
                    )

                # Rows - years
                # Columns -> features
                if timestep_predictions is None:
                    timestep_predictions = feature_prediction_df
                else:
                    # Concat by columns
                    timestep_predictions = pd.concat(
                        [timestep_predictions, feature_prediction_df], axis=1
                    )
            if prediction_df is None:
                prediction_df = timestep_predictions
            else:
                prediction_df = pd.concat([prediction_df, timestep_predictions], axis=0)

        return prediction_df


def train_models_for_ensemble_model(
    features: List[str],
) -> Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]]:
    # Basically function for training multiple models
    pass


if __name__ == "__main__":
    raise NotImplementedError(
        "Find out, whether it is useful to create an ensemble model!"
    )
