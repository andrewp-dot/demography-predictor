# Standard library imports
import os
import pandas as pd

from pydantic import BaseModel
from typing import Union, Optional, List, Any

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.utils.save_model import save_model, get_model
from src.preprocessors.data_transformer import DataTransformer

from src.base import TrainingStats

from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel

from src.global_model.model import GlobalModel

from src.predictors.predictor_base import DemographyPredictor


settings = Config()


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
        name: str,
        local_model_pipeline: LocalModelPipeline,
        global_model_pipeline: GlobalModelPipeline,
    ):
        # Name for storing the pipeline
        self.name: str = name

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
            input_data=scaled_data_df[FEATURES],
            last_year=last_year,
            target_year=target_year,
        )

        future_feature_values_scaled_df = pd.DataFrame(
            future_feature_values_scaled, columns=FEATURES
        )

        # Unscale data
        future_feature_values_df = self.local_model_pipeline.transformer.unscale_data(
            data=future_feature_values_scaled_df, columns=FEATURES
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
        # Create a new pipeline directory if does not exist

        trained_models_dir = os.path.abspath(settings.trained_models_dir)
        pipeline_dir = os.path.join(trained_models_dir, self.name)

        # Create pipeline dir if there is any
        if not os.path.isdir(pipeline_dir):
            os.makedirs(pipeline_dir)

        def save_to_pipeline_dir(model: Any, name: str):
            save_model(model=model, name=os.path.join(pipeline_dir, name))

        # Save local model and its transformer
        save_to_pipeline_dir(
            model=self.local_model_pipeline.model, name="local_model.pkl"
        )
        save_to_pipeline_dir(
            model=self.local_model_pipeline.transformer, name="local_transformer.pkl"
        )

        # Save global model and its transformer
        save_to_pipeline_dir(
            model=self.global_model_pipeline.model, name="global_model.pkl"
        )
        save_to_pipeline_dir(
            model=self.global_model_pipeline.transformer, name="global_transformer.pkl"
        )

    @classmethod
    def get_pipeline(cls, name: str):
        # Gets the pipeline by name

        trained_models_dir = os.path.abspath(settings.trained_models_dir)
        pipeline_dir = os.path.join(trained_models_dir, name)

        # Check if directory exist
        if not os.path.isdir(pipeline_dir):
            raise NotADirectoryError(
                f"Could not load pipeline '{name}. The pipeline directory ({pipeline_dir}) does not exist!"
            )

        def get_from_pipeline_dir(name: str) -> Any:
            return get_model(name=os.path.join(pipeline_dir, name))

        # Save local model and its transformer
        local_model = get_from_pipeline_dir(name="local_model.pkl")
        local_transformer = get_from_pipeline_dir(name="local_transformer.pkl")

        lm_pipeline = LocalModelPipeline(
            model=local_model, transformer=local_transformer
        )

        # Save global model and its transformer
        global_model = get_from_pipeline_dir(name="global_model.pkl")
        global_transformer = get_from_pipeline_dir(name="global_transformer.pkl")

        gm_pipeline = GlobalModelPipeline(
            model=global_model, transformer=global_transformer
        )

        return PredictorPipeline(
            name=name,
            local_model_pipeline=lm_pipeline,
            global_model_pipeline=gm_pipeline,
        )


def main():
    # Setup logging
    setup_logging()

    raise NotImplementedError("This pipeline is not implemented yet!")


if __name__ == "__main__":
    main()
