# Standard library imports
import pandas as pd

from pydantic import BaseModel
from typing import Union, Optional, List

# Custom imports
from src.utils.log import setup_logging
from src.preprocessors.data_transformer import DataTransformer

from src.base import TrainingStats

from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel

from src.global_model.model import GlobalModel

from src.predictors.predictor_base import DemographyPredictor


# !!!!!!! SCALER FOR INPUT DATA AND SCALER FOR THE TARGETS !!!!!!
# Ideas
# Local model pipeline ?

# Global model Pipeline?

# Is the DemographyPredictor the pipeline? I think YES


class LocalModelPipeline(BaseModel):
    model: Union[BaseLSTM, FineTunableLSTM, PureEnsembleModel]
    transformer: DataTransformer
    training_stats: Optional[TrainingStats] = None

    model_config = {"arbitrary_types_allowed": True}


class GlobalModelPipeline(BaseModel):
    model: GlobalModel
    transformer: DataTransformer

    model_config = {"arbitrary_types_allowed": True}


class PredictorPipeline:

    def __init__(
        self,
        local_model_pipeline: LocalModelPipeline,
        global_model_pipeline: GlobalModelPipeline,
    ):

        # Is this correct?
        self.local_model_pipeline = local_model_pipeline
        self.global_model_pipeline = global_model_pipeline

    def local_predict(
        self, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        FEATURES: List[str] = self.local_model_pipeline.model.FEATURES
        # Scale data
        scaled_data_df = self.local_model_pipeline.transformer.scale_data(
            data=input_data, columns=FEATURES
        )

        # Predict
        # Predict values to the future
        future_feature_values_scaled = self.local_model_pipeline.model.predict(
            input_data=scaled_data_df, last_year=last_year, target_year=target_year
        )

        # Unscale data
        future_feature_values_df = self.local_model_pipeline.transformer.unscale_data(
            data=future_feature_values_scaled, columns=FEATURES
        )

        return future_feature_values_df

    def global_predict(self, input_data: pd.DataFrame) -> pd.DataFrame:

        # Scale data
        FEATURES: List[str] = self.global_model_pipeline.model.FEATURES

        scaled_data = self.global_model_pipeline.transformer.scale_data(
            data=input_data, columns=FEATURES
        )

        # Predict
        predictions_df = self.global_model_pipeline.model.predict_human_readable(
            data=scaled_data
        )

        return predictions_df

    def predict(self, input_data: pd.DataFrame, target_year: int) -> pd.DataFrame:

        # Model predict
        # Get last known year
        if input_data.empty or not "year" in input_data.columns:
            raise ValueError(
                "Cannot find the last year of the data. Cannot find out the number of years to predict."
            )

        # Get last year
        last_year = int(
            input_data.sort_values(by="year", ascending=False)["year"].iloc[0].item()
        )

        future_feature_values_df = self.local_predict(
            input_data=input_data, last_year=last_year, target_year=target_year
        )

        # Adjust additional info for global model if needed
        if "country name" in self.global_model_pipeline.model.FEATURES:
            future_feature_values_df["country name"] = input_data["country name"][0]
            future_feature_values_df["country name"] = future_feature_values_df[
                "country name"
            ].astype("category")

        if "year" in self.global_model_pipeline.model.FEATURES:
            # Add year
            future_feature_values_df["year"] = pd.Series(
                list(range(last_year, target_year + 1))
            )

        # Adjust the order of the features
        future_feature_values_df = future_feature_values_df[
            self.global_model_pipeline.model.FEATURES
        ]

        predicted_data_df = self.global_predict(input_data=future_feature_values_df)

        # Return predictions
        return predicted_data_df

    def save_pipeline(self):
        raise NotImplementedError("Cannot save this pipeline yet.")


class DemographyPredictorPipeline:

    def __init__(self):
        raise NotImplementedError("This pipeline is not implemented yet!")


def main():
    # Setup logging
    setup_logging()

    raise NotImplementedError("This pipeline is not implemented yet!")


if __name__ == "__main__":
    main()
