# Standard library imports
import os
import pandas as pd
import numpy as np
import torch

from typing import Union, Optional, List, Any

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.utils.save_model import save_model, get_model
from src.preprocessors.data_transformer import DataTransformer


from src.base import TrainingStats


from src.local_model.model import RNNHyperparameters, BaseRNN
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel

from src.global_model.model import GlobalModel
from src.global_model.global_rnn import GlobalModelRNN
from src.global_model.statistical_wrapper import GlobalStatisticalWrapper


settings = Config()


class BasePipeline:

    def __init__(self, model: Any, transformer: DataTransformer, name: str):

        self.name: str = name
        self.model: Any = model
        self.transformer: DataTransformer = transformer

    def save_pipeline(self, custom_dir: Optional[str] = None):
        """
        Saves the model pipeline to default or custom_dir.

        Args:
            custom_dir (Optional[str], optional): If not specified, default directory is used. Defaults to None.
        """
        # Create a new pipeline directory if does not exist
        if custom_dir:
            trained_models_dir = custom_dir
        else:
            trained_models_dir = os.path.abspath(settings.trained_models_dir)

        pipeline_dir = os.path.join(trained_models_dir, self.name)

        # Create pipeline dir if there is any
        if not os.path.isdir(pipeline_dir):
            os.makedirs(pipeline_dir)

        def save_to_pipeline_dir(model: Any, name: str):
            save_model(model=model, name=os.path.join(pipeline_dir, name))

        # Save local model and its transformer
        save_to_pipeline_dir(model=self.model, name="model.pkl")
        save_to_pipeline_dir(model=self.transformer, name="transformer.pkl")

    @classmethod
    def get_pipeline(
        cls, name: str, custom_dir: Optional[str] = None
    ) -> "BasePipeline":
        # Gets the pipeline by name

        if custom_dir:
            trained_models_dir = custom_dir
        else:
            trained_models_dir = os.path.abspath(settings.trained_models_dir)
        """
        Get the model pipeline.

        Args:
            name (str): The pipeline name to get.
            custom_dir (Optional[str], optional): If not specified, default directory is used. Defaults to None.

        Raises:
            NotADirectoryError: Error if there is not found pipeline.

        Returns:
            out: BasePipeline: The desired pipeline of the specific pipeline type.
        """
        # Gets the pipeline by name
        pipeline_dir = os.path.join(trained_models_dir, name)

        # Check if directory exist
        if not os.path.isdir(pipeline_dir):
            raise NotADirectoryError(
                f"Could not load pipeline '{name}. The pipeline directory ({pipeline_dir}) does not exist!"
            )

        def get_from_pipeline_dir(name: str) -> Any:
            return get_model(name=os.path.join(pipeline_dir, name))

        # Save local model and its transformer
        model = get_from_pipeline_dir(name="model.pkl")
        transformer = get_from_pipeline_dir(name="transformer.pkl")

        return cls(name=name, model=model, transformer=transformer)


class LocalModelPipeline(BasePipeline):

    # Set the correct typehints
    model: Union["BaseRNN", "FineTunableLSTM", "PureEnsembleModel"]
    training_stats: Optional["TrainingStats"]

    def __init__(
        self,
        model: Union["BaseRNN", "FineTunableLSTM", "PureEnsembleModel"],
        transformer: DataTransformer,
        name: str = "local_model_pipeline",
        training_stats: Optional["TrainingStats"] = None,
    ):
        super(LocalModelPipeline, self).__init__(
            model=model, transformer=transformer, name=name
        )
        self.training_stats = training_stats

    def predict(
        self, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:
        FEATURES: List[str] = self.model.FEATURES
        TARGETS: List[str] = self.model.TARGETS

        # Scale data

        # If it is pure ensemble model and transformer is None, it is supposed that it is ensemble local arima model
        SUPPOSED_ARIMA_MODEL: bool = False
        if (
            isinstance(self.model, PureEnsembleModel)
            and self.transformer.SCALER is None
        ):
            scaled_data_df = input_data
            SUPPOSED_ARIMA_MODEL = True

        elif FEATURES == TARGETS:
            scaled_data_df = self.transformer.scale_data(
                data=input_data,
                features=FEATURES,
            )
        else:
            scaled_data_df = self.transformer.scale_data(
                data=input_data, features=FEATURES, targets=TARGETS
            )

        # Predict
        # Predict values to the future

        # if isinstance(self.model, ExpLSTM):
        #     future_feature_values_scaled = self.__experimental_model_predict(
        #         state_data=scaled_data_df[FEATURES],
        #         last_year=last_year,
        #         target_year=target_year,
        #     )
        # else:

        future_feature_values_scaled = self.model.predict(
            input_data=scaled_data_df[FEATURES],
            last_year=last_year,
            target_year=target_year,
        )

        future_feature_values_scaled_df = pd.DataFrame(
            future_feature_values_scaled, columns=FEATURES
        )

        # Unscale targets
        if SUPPOSED_ARIMA_MODEL:
            future_feature_values_df = future_feature_values_scaled_df
        else:
            future_feature_values_df = self.transformer.unscale_data(
                data=future_feature_values_scaled_df, targets=TARGETS
            )

        return future_feature_values_df


class GlobalModelPipeline(BasePipeline):

    def __init__(
        self,
        model: GlobalModel,
        transformer: DataTransformer,
        name: str = "global_model_pipeline",
        is_statistical_model: bool = False,
    ):
        self.name: str = name
        self.model: Union[GlobalModel, GlobalModelRNN] = model
        self.transformer: DataTransformer = transformer

        # This flag is used to detect whether need to or does not need to scale data in predict method
        self.IS_STATISTICAL_MODEL: bool = is_statistical_model

    def __tree_based_model_predict(
        self, input_data_batch: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Based on provided input data generates predictions for the tree based model.

        Args:
            input_data_batch (pd.DataFrame): Scaled input data.

        Returns:
            out: pd.DataFrame: Predicted data based on input data
        """
        return self.model.predict_human_readable(data=input_data_batch)

    def __rnn_based_model_predict(self, input_data_batch: pd.DataFrame) -> torch.Tensor:
        """
        Based on provided input data generates predictions for the RNN based model.

        Args:
            input_data_batch (pd.DataFrame): Scaled input data.

        Returns:
            out: pd.DataFrame: Predicted data based on input data
        """
        # Shape: (samples, timesteps, target_num)
        scaled_predictions = self.model.predict(input_data=input_data_batch)

        # Need to reshape to (samples, target_num)
        scaled_predictions = scaled_predictions.reshape(
            scaled_predictions.shape[0],
            scaled_predictions.shape[1] * scaled_predictions.shape[2],
        )

        return scaled_predictions

    def __statistical_model_predict(
        self, state: str, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        return self.model.predict(
            state=state,
            input_data=input_data,
            last_year=last_year,
            target_year=target_year,
        )

    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
        state: Optional[str] = None,
    ) -> pd.DataFrame:
        iterations_num = target_year - last_year

        FEATURES = self.model.FEATURES
        TARGETS = self.model.TARGETS

        # Scale features
        TO_SCALE_TARGETS = TARGETS
        # If there is tree based model -> do not scale the targets
        if isinstance(self.model, GlobalModel):
            TO_SCALE_TARGETS = None

        # Do not scale targets for statistial model
        if isinstance(self.model, GlobalStatisticalWrapper):
            scaled_data_df = input_data
        else:
            scaled_data_df = self.transformer.scale_data(
                data=input_data, features=FEATURES, targets=TO_SCALE_TARGETS
            )

        # If the model is statisical, model
        if isinstance(self.model, GlobalStatisticalWrapper):

            if state is None:
                raise ValueError(
                    "For the GlobalModelPipeline with model of type 'GlobalStatisticalWrapper' need state to be provided as argument!"
                )

            return self.__statistical_model_predict(
                state=state,
                input_data=scaled_data_df,
                last_year=last_year,
                target_year=target_year,
            )

        # Convert to numpy
        features_np = scaled_data_df[FEATURES].values  # shape (time, num_features)
        targets_np = scaled_data_df[TARGETS].values  # shape (time, num_targets)

        # Get the known targets -> should be nan value in the input data
        targets_for_pred = targets_np[:-iterations_num]

        # Prepare array to collect predictions
        all_predictions = [targets_for_pred.copy()]  # list of arrays

        for i in range(iterations_num):
            # Build input features (current features + last targets)
            # Assume your model can handle concatenated features

            current_features = features_np[: -(iterations_num - i)]

            input_batch = np.concatenate([current_features, targets_for_pred], axis=1)
            input_batch_df = pd.DataFrame(
                input_batch,
                columns=self.model.FEATURES
                + self.model.TARGETS,  # <- concat of feature and target names
            )

            # Predict next target
            if isinstance(self.model, GlobalModel):
                next_target_preds = self.__tree_based_model_predict(
                    input_data_batch=input_batch_df
                )

                # Convert to tensor
                next_target_preds = next_target_preds.values  # (samples, target_num)

            elif isinstance(self.model, GlobalModelRNN):
                next_target_preds = self.__rnn_based_model_predict(
                    input_data_batch=input_batch_df
                )

            else:
                raise ValueError(
                    f"Global model pipeline predict does not support predict of model type: {type(self.model).__name__}"
                )

            # Stack predicted next steps

            targets_for_pred = np.vstack(
                [targets_for_pred, next_target_preds[-1:]]
            )  # only last predicton
            all_predictions.append(next_target_preds[-1:])

        # Stack all predicted targets
        final_predictions = np.vstack(all_predictions)

        predictions_df = pd.DataFrame(final_predictions, columns=TARGETS)

        # Unscale it here
        # Unscale
        if TO_SCALE_TARGETS:
            unscaled_predictions = self.transformer.unscale_data(
                data=pd.DataFrame(predictions_df, columns=TO_SCALE_TARGETS),
                targets=TO_SCALE_TARGETS,
            )

            predictions_df = unscaled_predictions

        # Return only last predictions
        return predictions_df.tail(iterations_num).reset_index(drop=True)


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
        if "country_name" in self.global_model_pipeline.model.FEATURES:
            future_feature_values_df["country_name"] = input_data["country_name"][0]
            future_feature_values_df["country_name"] = future_feature_values_df[
                "country_name"
            ].astype("category")

        if "year" in self.global_model_pipeline.model.FEATURES:
            # Add year
            future_feature_values_df["year"] = pd.Series(
                list(range(last_year, target_year + 1))
            )

        # Adjust the order of the features
        # TODO: concat future_feature_values with previous target values
        future_feature_values_df = future_feature_values_df[
            self.global_model_pipeline.model.FEATURES
        ]

        ### CHANGE THIS
        # Append the values of the future predictions to the input data
        future_feature_values_df = pd.concat(
            [
                input_data[self.global_model_pipeline.model.FEATURES],
                future_feature_values_df,
            ],
            axis=0,
        )

        # Get history of the previous targets
        previous_targets_df = input_data[self.global_model_pipeline.model.TARGETS]

        # Pad previous_targets_df to match the length
        pad_len = len(future_feature_values_df) - len(previous_targets_df)

        # Append nan for unknown -> to predict values
        padding = pd.DataFrame(
            np.nan, columns=previous_targets_df.columns, index=range(pad_len)
        )

        previous_targets_padded = pd.concat(
            [previous_targets_df, padding], ignore_index=True
        )

        # Get the final input data
        final_input = pd.concat(
            [
                future_feature_values_df.reset_index(drop=True),
                previous_targets_padded.reset_index(drop=True),
            ],
            axis=1,
        )

        predicted_data_df = self.global_model_pipeline.predict(
            input_data=final_input,
            last_year=last_year,
            target_year=target_year,
        )

        # Add years to final predictions
        PREDICTED_YEARS = list(range(last_year + 1, target_year + 1))
        predicted_years_df = pd.DataFrame({"years": PREDICTED_YEARS})

        years_final_predictions_df = pd.concat(
            [predicted_years_df, predicted_data_df], axis=1
        )

        return years_final_predictions_df

    def save_pipeline(self, custom_dir: Optional[str] = None):
        # Create a new pipeline directory if does not exist
        if custom_dir:
            trained_models_dir = custom_dir
        else:
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
    def get_pipeline(cls, name: str, custom_dir: Optional[str] = None):
        # Gets the pipeline by name

        if custom_dir:
            trained_models_dir = custom_dir
        else:
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
