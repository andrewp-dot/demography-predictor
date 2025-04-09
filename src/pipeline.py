# Standard library imports
import pandas as pd

# Custom imports
from src.utils.log import setup_logging
from preprocessors.data_transformer import DataTransformer

from src.predictors.predictor_base import DemographyPredictor


# !!!!!!! SCALER FOR INPUT DATA AND SCALER FOR THE TARGETS !!!!!!
# Ideas
# Local model pipeline ?

# Global model Pipeline?

# Is the DemographyPredictor the pipeline? I think YES


class PredictorPipeline:

    def __init__(self, preprocessor: DataTransformer, model: DemographyPredictor):

        self.preprocessor: DataTransformer = preprocessor
        self.model: DemographyPredictor = model

        # Is this correct?
        self.local_model = model.local_model
        self.global_model = model.global_model
        raise NotImplementedError("This pipeline is not implemented yet!")

    def predict(self, input_data: pd.DataFrame, target_year: int):
        # !!!!!!! SCALER FOR INPUT DATA AND SCALER FOR THE TARGETS !!!!!!

        # Scale data
        scaled_data_df = self.preprocessor.scale_data(
            data=input_data,
            columns=self.model.FEATURES,
            scaler=self.preprocessor.SCALER,
        )

        # Model predict
        # Get last known year

        if input_data.empty or not "year" in input_data.columns:
            raise ValueError(
                "Cannot find the last year of the data. Cannot find out the number of years to predict."
            )

        last_year = int(
            input_data.sort_values(by="year", ascending=False)["year"].iloc[0].item()
        )

        predicted_data_df = self.model.predict(
            input_data=scaled_data_df, last_year=last_year, target_year=target_year
        )

        # Unscale data
        self.preprocessor.unscale_data(
            data=predicted_data_df, columns=self.model.TARGETS
        )

        # Return predictions


def main():
    # Setup logging
    setup_logging()

    raise NotImplementedError("This pipeline is not implemented yet!")


if __name__ == "__main__":
    main()
