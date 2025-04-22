# Standard library
import shap
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import os

import pandas as pd

from typing import List

# Filters about rng seed warning - yet the shap does not acceps 'rng' argument.
import warnings

# Custom imports
from src.utils.save_model import get_model
from src.pipeline import LocalModelPipeline, GlobalModelPipeline

from src.local_model.model import BaseLSTM
from src.local_model.experimental import ExpLSTM
from src.preprocessors.multiple_states_preprocessing import StateDataLoader
from src.preprocessors.data_transformer import DataTransformer


from src.utils.log import setup_logging


warnings.filterwarnings("ignore", message=".*The NumPy global RNG was seeded.*")


logger = logging.getLogger("method_selection")


class LSTMExplainer:

    class ReducedModelWrapper(torch.nn.Module):
        def __init__(self, model, predicted_timestep_index: int = 0):
            super().__init__()
            self.model = model
            self.predicted_timestep_index = predicted_timestep_index

        def forward(self, x):
            out = self.model(x)
            return out[:, self.predicted_timestep_index, :]

    def __init__(self, pipeline: LocalModelPipeline):
        self.pipeline: LocalModelPipeline = pipeline
        # Get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_sequences(self, state: str) -> torch.Tensor:

        # Load data
        STATE: str = state

        loader = StateDataLoader(state=STATE)
        states_data = loader.load_data()

        train_df, test_df = loader.split_data(
            data=states_data,
            split_rate=0.8,
        )

        # Scale data
        FEATURES = self.pipeline.model.FEATURES
        TARGETS = self.pipeline.model.TARGETS

        if FEATURES == TARGETS:
            TARGETS = None

        scaled_state_df = self.pipeline.transformer.scale_data(
            data=train_df,
            features=FEATURES,
            targets=TARGETS,
        )

        # Create sequences
        input_sequences = self.pipeline.transformer.create_sequences(
            input_data=scaled_state_df,
            columns=self.pipeline.model.FEATURES,
            sequence_len=self.pipeline.model.hyperparameters.sequence_length,
        )

        # print(scaled_state_df)

        # print(input_sequences.shape)

        input_sequences.requires_grad = True
        return input_sequences

    def get_shap_values(
        self, input_sequences: torch.Tensor, predicted_timestep: int = 0
    ) -> torch.Tensor:
        """
        Compute SHAP values for the given input sequences.

        Args:
            input_sequences (torch.Tensor): Input sequences for computing the shap values.
            predicted_timestep (int): The predicted timestep to compute values for (if the model predicts values for multiple timesteps). Defaults to 0.

        Returns:
            out: torch.Tensor: The shap values.
        """

        # Put the model to the device
        self.pipeline.model.to(device=self.device)

        # Set training batches as input_x

        self.pipeline.model.train()
        self.pipeline.model.lstm.train()

        # Get the model to the wrapper to get the time step output of the feature in index
        model = self.ReducedModelWrapper(
            model=self.pipeline.model,
            predicted_timestep_index=predicted_timestep,
        )

        # # Set model to evaluation mode
        # self.pipeline.model.eval()

        # Compute shap values
        torch.backends.cudnn.enabled = False

        # Move input sequences to the same device as the model
        input_sequences = input_sequences.to(device=self.device)

        # Compute SHAP values
        explainer = shap.GradientExplainer(model, input_sequences)
        shap_values = explainer.shap_values(input_sequences)

        print(f"Shap vales shape: {shap_values.shape}")

        torch.backends.cudnn.enabled = True

        return shap_values

    def get_feature_importance(
        self,
        shap_values: torch.Tensor,
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> dict:
        # Aggregate SHAP values
        shap_array = np.array(shap_values)

        # Assuming your task is sequence-based, we’ll average across time steps (axis=1) and instances (axis=0)
        # mean_shap = np.abs(shap_values[0]).mean(axis=(0, 1))

        mean_shap = np.abs(shap_array).mean(axis=(0, 1, 3))

        # Get feature names from model
        feature_names = self.pipeline.model.FEATURES  # assuming it's a list of strings

        # Create and sort dict by vylues
        feature_importance_dict = {
            fname: value for fname, value in zip(feature_names, mean_shap)
        }

        # Sort the dictionary by values in descending order
        sorted_feature_importance_dict = dict(
            sorted(feature_importance_dict.items(), key=lambda item: item[1])
        )

        ## FEATURE IMPORTANCE PLOT
        plt.figure(figsize=(30, 5))
        plt.barh(
            list(sorted_feature_importance_dict.keys()),
            list(sorted_feature_importance_dict.values()),
        )
        plt.ylabel("Features")
        plt.xlabel("Mean |SHAP Value| (Feature Importance)")
        plt.title("Overall Feature Importance")
        # plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            save_path = os.path.join(save_path, "lstm_shap_feature_explanation_fig.png")
            plt.savefig(save_path)

        if show_plot:
            plt.show()

        return sorted_feature_importance_dict

    def get_force_plot(
        self,
        shap_values: torch.Tensor,
        input_sequences: torch.Tensor,
        save_path: str | None = None,
        sample_idx: int = 0,
        time_step: int = 0,
        show_plot: bool = False,
    ) -> None:
        # Get the SHAP values and input for that time step
        shap_vals_single = shap_values[0][sample_idx, time_step]
        input_single = input_sequences[sample_idx, time_step].detach().cpu().numpy()

        # Plot the force plot
        shap.force_plot(
            base_value=0,  # If your model has no baseline, 0 is fine
            shap_values=shap_vals_single,
            features=input_single,
            feature_names=self.pipeline.model.FEATURES,
            matplotlib=True,
            show=False,
            figsize=(50, 5),
            contribution_threshold=0.1,
        )

        if save_path:
            save_path = os.path.join(save_path, "lstm_shap_force_plot.png")

            # Save the force plot to a file (using matplotlib)
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

    def get_summary_plot(
        self,
        shap_values: torch.Tensor,
        input_x: torch.Tensor,
        save_path: str | None = None,
        sample_idx: int = 0,
        target_index: int = 0,
        show_plot: bool = False,
    ) -> None:

        print("Summary plot...")

        input_np = input_x.detach().cpu().numpy()
        input_x_single = input_np[
            sample_idx, :, :
        ]  # Shape: (12, 13) - Take the first batch (12 time steps, 13 features)

        # Flatten for plotting
        input_x_single_flat = input_x_single.reshape(
            -1, input_x_single.shape[-1]
        )  # Shape: (12 * 13, 13)

        # Similarly, flatten SHAP values for the same batch
        shap_vals_single = shap_values[target_index][
            :, sample_idx, :
        ]  # Shape: (12, 13) - Take the first batch, all time steps
        shap_vals_single_flat = shap_vals_single.reshape(
            -1, shap_vals_single.shape[-1]
        )  # Shape: (12 * 13, 13)

        # Plot the summary plot
        shap.summary_plot(
            shap_vals_single_flat,  # Flattened SHAP values for the first batch
            input_x_single_flat,  # Flattened input features for the first batch
            feature_names=self.pipeline.model.FEATURES,  # List of feature names
            show=False,
        )

        plt.tight_layout()

        if save_path:

            save_path = os.path.join(save_path, "lstm_shap_summary_plot.png")

            # Save the force plot to a file (using matplotlib)
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

    def get_waterfall_plot(
        self,
        shap_values: torch.Tensor,
        input_x: torch.Tensor,
        save_path: str | None = None,
        sample_idx: int = 0,
        target_index: int = 0,
        time_step_idx: int = 0,
        show_plot: bool = False,
    ) -> None:
        print("Waterfall plot...")

        shap_values_for_instance = shap_values[target_index][:, sample_idx, :]

        input_np = input_x.detach().cpu().numpy()
        input_x_single = input_np[sample_idx, :, :]

        # We need the input features for the instance
        input_features_for_instance = input_x_single[sample_idx, :]

        output_idx = 0

        base_value = self.pipeline.model.shap_predict(
            input_x[sample_idx : sample_idx + 1].to(self.device)
        )  # base_value shape (from forward method): (1, future_time_step_predicted, feature_num)

        # Extract desired feature
        desired_feature_value = base_value[0, 0, output_idx].item()

        # Create the waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_for_instance[
                    time_step_idx
                ],  # SHAP values for the instance
                base_values=desired_feature_value,
                data=input_features_for_instance,  # The input features
                feature_names=self.pipeline.model.FEATURES,  # Feature names
            ),
            max_display=10,
        )

        # Save the waterfall plot

        if save_path:

            save_path = os.path.join(save_path, "lstm_shap_waterfall_plot.png")

            # Save the force plot to a file (using matplotlib)
            fig = plt.gcf()  # Get current figure
            fig.set_size_inches(10, 5)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.9)
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")


class GlobalModelExplainer:

    class MultiOutputWrapper:
        def __init__(self, model):
            self.model = model

        def __call__(self, X):
            return self.model.predict_human_readable(X)

        def shap_predict(self, X):
            # For SHAP, we need to also return the base value for each output
            # Typically, for multi-output models, this would be the mean prediction across all instances.
            # You can also use model's base prediction if it provides one.

            predictions = self.model.predict(X)  # Model's predictions for each output
            base_values = np.mean(predictions, axis=0)  # This is a simplified approach

            return predictions, base_values

    def __init__(self, pipeline: GlobalModelPipeline):
        self.pipeline: GlobalModelPipeline = pipeline
        self.FEATURE_NAMES: List[str] = (
            self.pipeline.model.FEATURES + self.pipeline.model.HISTORY_TARGET_FEATURES
        )

        self.model = self.pipeline.model

    def create_inputs(self, state: str) -> pd.DataFrame:

        # Load data
        loader = StateDataLoader(state=state)
        input_data = loader.load_data()

        # Create input
        input_dict, _ = self.pipeline.model.create_state_inputs_outputs(
            states_dict={state: input_data}
        )

        # Create tensor?
        return input_dict[state]

    def get_shap_values(self, X: pd.DataFrame):

        # Wrap this in here because shap cannot handle multioutput regressor
        model_wrapper = self.MultiOutputWrapper(model=self.model)
        explainer = shap.Explainer(model_wrapper, X)
        shap_values = explainer(X)
        return shap_values

    def get_feature_importance(
        self,
        shap_values: shap.Explanation,
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> dict:

        # Convert shap explanation object to numpy array
        shap_values_array = (
            shap_values.values
        )  # Shappe: (samples_num, features_num, targets_num)

        # Aggregate SHAP values -> accross samples and features
        mean_shap = np.abs(shap_values_array).mean(axis=(0, 2))

        print(mean_shap)

        feature_names = self.FEATURE_NAMES

        # Create and sort dict by vylues
        feature_importance_dict = {
            fname: value for fname, value in zip(feature_names, mean_shap)
        }

        # Sort the dictionary by values in descending order
        sorted_feature_importance_dict = dict(
            sorted(
                feature_importance_dict.items(), key=lambda item: item[1], reverse=True
            )
        )

        # import pprint

        # pprint.pprint(sorted_feature_importance_dict)

        ## FEATURE IMPORTANCE PLOT
        plt.figure(figsize=(10, 10))
        plt.barh(
            list(sorted_feature_importance_dict.keys()),
            list(sorted_feature_importance_dict.values()),
        )
        plt.ylabel("Features")
        plt.xlabel("Mean |SHAP Value| (Feature Importance)")
        plt.title("Overall Feature Importance")
        # plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:

            save_path = os.path.join(save_path, "gm_shap_waterfall_plot.png")
            plt.savefig(save_path)

        if show_plot:
            plt.show()

        return sorted_feature_importance_dict


def main():
    # Setup logging
    setup_logging()

    # Get pipeline
    pipeline = LocalModelPipeline(
        # model=get_model("core_pipeline/local_model.pkl"),
        # transformer=get_model("core_pipeline/local_transformer.pkl"),
        model=get_model("ExpLSTM_pop_total.pkl"),
        transformer=get_model("ExpLSTM__pop_total_transformer.pkl"),
    )

    explainer = LSTMExplainer(pipeline=pipeline)

    # Get input and shap values
    input_x = explainer.create_sequences(state="Czechia")
    shap_values = explainer.get_shap_values(input_sequences=input_x)

    # Get feature importance
    feature_importance = explainer.get_feature_importance(
        shap_values=shap_values, show_plot=True, save_plot=True
    )
    logger.info(f"Feature importance: {feature_importance}")

    print("Creating plots...")
    explainer.get_force_plot(
        shap_values=shap_values,
        input_sequences=input_x,
        sample_idx=0,
        time_step=0,
        save_plot=True,
    )
    explainer.get_summary_plot(
        shap_values=shap_values,
        input_x=input_x,
        sample_idx=0,
        target_index=0,
        save_plot=True,
    )

    # TODO: fix this
    explainer.waterfall_plot(
        shap_values=shap_values,
        input_x=input_x,
        sample_idx=0,
        target_index=0,
        save_plot=True,
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()
