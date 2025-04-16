# Standard library imports
import os
import pandas as pd

from typing import Union, Optional, List, Any

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.utils.save_model import save_model, get_model
from src.preprocessors.data_transformer import DataTransformer


from src.base import TrainingStats

from src.local_model.experimental import ExpLSTM
from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel

from src.global_model.model import GlobalModel


settings = Config()


class LocalModelPipeline:

    def __init__(
        self,
        model: Union[BaseLSTM, FineTunableLSTM, PureEnsembleModel],
        transformer: DataTransformer,
        name: str = "local_model_pipeline",
        training_stats: Optional[TrainingStats] = None,
    ):
        self.name: str = name
        self.model: Union[BaseLSTM, FineTunableLSTM, PureEnsembleModel] = model
        self.transformer: DataTransformer = transformer
        self.training_stats: Optional[TrainingStats] = training_stats

    def __experimental_model_predict(
        self, state_data: pd.DataFrame, last_year: int, target_year: int
    ):

        # print(state_data)

        model: ExpLSTM = self.model

        FEATURES: List[str] = self.model.FEATURES
        TARGETS: List[str] = self.model.TARGETS
        SEQUENCE_LEN: int = self.model.hyperparameters.sequence_length

        # Note: the all data should contain only input features
        if set(state_data.columns) != set(FEATURES):
            raise ValueError(
                f"The input data columns ({set(state_data.columns)})  are not compatibile with the model features: ({set(FEATURES)})"
            )

        to_predict_years_num = target_year - last_year

        to_predict_years_num = (
            target_year - last_year
        )  # To include also the target year

        # Get the future values of non-target features
        NON_TARGET_FEATURES = [f for f in FEATURES if not f in TARGETS]
        all_non_target_values_df = state_data[NON_TARGET_FEATURES]

        # Create N sequences needed for prediction using non target feature true values

        # Here is the error
        all_known_target_values_df = state_data.iloc[
            -(to_predict_years_num + SEQUENCE_LEN) : -to_predict_years_num
        ][TARGETS]

        # Save the order

        def rolling_window(
            non_target_df: pd.DataFrame, target_df: pd.DataFrame, offset: int
        ) -> pd.DataFrame:

            # Offset is here due to years

            # Get the last sequence of non target dataframe
            non_target_df = non_target_df.iloc[
                -(offset + SEQUENCE_LEN) : -offset
            ].reset_index(drop=True)

            target_df = target_df.tail(SEQUENCE_LEN).reset_index(drop=True)

            input_data: pd.DataFrame = pd.concat([non_target_df, target_df], axis=1)[
                FEATURES
            ]

            return input_data

        input_data: pd.DataFrame = rolling_window(
            non_target_df=all_non_target_values_df,
            target_df=all_known_target_values_df,
            offset=to_predict_years_num,
        )

        for offset in range(to_predict_years_num, 0, -1):

            next_year_targets = model.predict(input_data=input_data)

            # Create a dataframe from it
            next_year_targets_df = pd.DataFrame(next_year_targets, columns=TARGETS)

            # Add predictions to the existing target data
            all_known_target_values_df = pd.concat(
                [all_known_target_values_df, next_year_targets_df],
                axis=0,
            )

            # Update input data
            input_data = rolling_window(
                non_target_df=all_non_target_values_df,
                target_df=all_known_target_values_df,
                offset=offset,
            )

        # Return the last N predictions
        predictions_df = all_known_target_values_df.tail(to_predict_years_num)[TARGETS]

        return predictions_df

    def predict(
        self, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:
        FEATURES: List[str] = self.model.FEATURES
        TARGETS: List[str] = self.model.TARGETS

        # Scale data
        scaled_data_df = self.transformer.scale_data(
            data=input_data, features=FEATURES, targets=TARGETS
        )

        # Predict
        # Predict values to the future

        if isinstance(self.model, ExpLSTM):
            future_feature_values_scaled = self.__experimental_model_predict(
                state_data=scaled_data_df[FEATURES],
                last_year=last_year,
                target_year=target_year,
            )

        else:
            future_feature_values_scaled = self.model.predict(
                input_data=scaled_data_df[FEATURES],
                last_year=last_year,
                target_year=target_year,
            )

        future_feature_values_scaled_df = pd.DataFrame(
            future_feature_values_scaled, columns=FEATURES
        )

        # Unscale targets
        future_feature_values_df = self.transformer.unscale_data(
            data=future_feature_values_scaled_df, targets=TARGETS
        )

        return future_feature_values_df


class GlobalModelPipeline:
    # model: GlobalModel
    # transformer: DataTransformer

    # model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        model: GlobalModel,
        transformer: DataTransformer,
        name: str = "global_model_pipeline",
    ):
        self.name: str = name
        self.model: GlobalModel = model
        self.transformer: DataTransformer = transformer

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:

        # Scale data
        FEATURES: List[str] = self.model.FEATURES

        scaled_data = self.transformer.scale_data(data=input_data, columns=FEATURES)

        # Predict
        predictions_df = self.model.predict_human_readable(data=scaled_data)

        return predictions_df


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

        future_feature_values_df = self.local_model_pipeline.predict(
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

        predicted_data_df = self.global_model_pipeline.predict(
            input_data=future_feature_values_df
        )

        # Add years to final predictions
        PREDICTED_YEARS = list(range(last_year + 1, target_year + 1))
        predicted_years_df = pd.DataFrame({"years": PREDICTED_YEARS})

        years_final_predictions_df = pd.concat(
            [predicted_years_df, predicted_data_df], axis=1
        )

        return years_final_predictions_df

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
