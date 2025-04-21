# Standard library
import shap
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging

import pandas as pd

from typing import Dict

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


warnings.filterwarnings("ignore", category=FutureWarning, module="shap")


logger = logging.getLogger("method_selection")


class LSTMExplainer:

    def __init__(self, pipeline: LocalModelPipeline):
        self.pipeline: LocalModelPipeline = pipeline

    def create_sequences(self, state: str) -> torch.Tensor:

        # Load data
        STATE: str = state

        loader = StateDataLoader(state=STATE)
        states_data = loader.load_data()

        train_df, test_df = loader.split_data(
            data=states_data,
            split_rate=0.8,
        )

        print(self.pipeline.model.FEATURES)

        # Scale data
        scaled_state_df = self.pipeline.transformer.scale_data(
            data=train_df,
            features=self.pipeline.model.FEATURES,
            targets=self.pipeline.model.TARGETS,
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

    def get_shap_values(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute SHAP values for the given input sequences.
        """

        # Put the model to the device
        self.pipeline.model.to(device=self.pipeline.model.device)

        # Set training batches as input_x
        self.pipeline.model.train()
        self.pipeline.model.lstm.train()

        # # Set model to evaluation mode
        # self.pipeline.model.eval()

        # Compute shap values
        torch.backends.cudnn.enabled = False

        # Move input sequences to the same device as the model
        input_sequences = input_sequences.to(device=self.pipeline.model.device)

        # Compute SHAP values
        explainer = shap.GradientExplainer(self.pipeline.model, input_sequences)
        shap_values = explainer.shap_values(input_sequences)

        print("SHAP VALUES: ")
        print(shap_values.shape)

        torch.backends.cudnn.enabled = True

        return shap_values

    def get_feature_importance(
        self,
        shap_values: torch.Tensor,
        show_plot: bool = False,
        save_plot: bool = False,
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
        plt.figure(figsize=(10, 5))
        plt.barh(
            list(sorted_feature_importance_dict.keys()),
            list(sorted_feature_importance_dict.values()),
        )
        plt.ylabel("Features")
        plt.xlabel("Mean |SHAP Value| (Feature Importance)")
        plt.title("Overall Feature Importance")
        # plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plot:
            plt.savefig("shap_explanation_fig.png")

        if show_plot:
            plt.show()

        return sorted_feature_importance_dict

    def get_force_plot(
        self,
        shap_values: torch.Tensor,
        input_sequences: torch.Tensor,
        sample_idx: int = 0,
        time_step: int = 0,
        save_plot: bool = False,
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
        )

        if save_plot:
            # Save the force plot to a file (using matplotlib)
            plt.savefig(
                "shap_force_plot.png", format="png", dpi=300, bbox_inches="tight"
            )

        if show_plot:
            plt.show()

    def get_summary_plot(
        self,
        shap_values: torch.Tensor,
        input_x: torch.Tensor,
        sample_idx: int = 0,
        target_index: int = 0,
        show_plot: bool = False,
        save_plot: bool = False,
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

        if save_plot:
            plt.savefig(
                "shap_summary_plot.png", format="png", dpi=300, bbox_inches="tight"
            )

        if show_plot:
            plt.show()

    def waterfall_plot(
        self,
        shap_values: torch.Tensor,
        input_x: torch.Tensor,
        sample_idx: int = 0,
        target_index: int = 0,
        show_plot: bool = False,
        save_plot: bool = False,
    ) -> None:
        print("Waterfall plot...")

        shap_values_for_instance = shap_values[target_index][
            :, sample_idx, :
        ]  # Shape: (12, 13) - Take the first batch, all time steps

        input_np = input_x.detach().cpu().numpy()
        input_x_single = input_np[sample_idx, :, :]

        # We need the input features for the instance
        input_features_for_instance = input_x_single[sample_idx, :]

        print(input_x[sample_idx : sample_idx + 1].shape)

        output_idx = 0

        # TODO: fix this to use just the predict
        base_value = self.pipeline.model.shap_predict(
            input_x[sample_idx : sample_idx + 1]
        )[0, output_idx].item()

        # Create the waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_for_instance,  # SHAP values for the instance
                base_values=base_value,
                data=input_features_for_instance,  # The input features
                feature_names=self.pipeline.model.FEATURES,  # Feature names
            )
        )

        # Save the waterfall plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.9)
        plt.savefig(
            "shap_waterfall_plot.png", format="png", dpi=300, bbox_inches="tight"
        )


class GlobalModelExplainer:

    def __init__(self, pipeline: GlobalModelPipeline):
        self.pipeline: GlobalModelPipeline = pipeline

    def create_inputs(self) -> pd.DataFrame:
        raise NotImplementedError("Not implemented yet.")

    def get_feature_importance(
        self,
        shap_values: torch.Tensor,
        show_plot: bool = False,
        save_plot: bool = False,
    ) -> dict:
        # Aggregate SHAP values
        # Assuming your task is sequence-based, we’ll average across time steps (axis=1) and instances (axis=0)
        mean_shap = np.abs(shap_values[0]).mean(axis=(0, 1))

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
        plt.figure(figsize=(10, 5))
        plt.barh(
            list(sorted_feature_importance_dict.keys()),
            list(sorted_feature_importance_dict.values()),
        )
        plt.ylabel("Features")
        plt.xlabel("Mean |SHAP Value| (Feature Importance)")
        plt.title("Overall Feature Importance")
        # plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plot:
            plt.savefig("shap_explanation_fig.png")

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
