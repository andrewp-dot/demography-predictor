import shap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.save_model import get_model

from src.local_model.model import BaseLSTM
from src.preprocessors.multiple_states_preprocessing import StateDataLoader
from src.preprocessors.data_transformer import DataTransformer

from src.utils.log import setup_logging

# Filters about rng seed warning - yet the shap does not acceps 'rng' argument.
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="shap")


# Setup logging
setup_logging()


# Get model and transformer
model: BaseLSTM = get_model(name="BaseLSTM.pkl")
transformer: DataTransformer = get_model("BaseLSTM_transformer.pkl")
# model.to(model.device)

# Get input data
INPUT_STATE_DATA: str = "Czechia"
loader = StateDataLoader(state=INPUT_STATE_DATA)

states_data = loader.load_data()

train_df, test_df = loader.split_data(
    data=states_data,
    split_rate=0.8,
)

LAST_YEAR: int = loader.get_last_year(train_df)
TARGET_YEAR: int = loader.get_last_year(test_df)

print(f"Sequence length: {model.hyperparameters.sequence_length}")

# Scale data
scaled_state_df = transformer.scale_data(data=states_data, columns=model.FEATURES)


# Create sequences
def create_sequences() -> torch.Tensor:

    input_sequences = transformer.create_sequences(
        input_data=scaled_state_df,
        columns=model.FEATURES,
        sequence_len=model.hyperparameters.sequence_length,
    )

    # Save input sequences
    # Extend instead of append to flatten
    return input_sequences


# Setup sequences
input_sequences = create_sequences()
input_sequences.requires_grad = True

print(f"Nnmber of input sequences: {len(input_sequences)}")


# Set training batches as input_x
model.train()
model.lstm.train()


model.to(device=model.device)
input_x = input_sequences.to(device=model.device)

print("\n----- Shapes".ljust(100, "-"))
print(f"input_X shape: {input_x.shape}")

# Compute shap values
torch.backends.cudnn.enabled = False
# with torch.enable_grad():
explainer = shap.GradientExplainer(model, input_x)
# Compute SHAP values
shap_values = explainer.shap_values(input_x)
torch.backends.cudnn.enabled = True
print(f"shap_values[0] (from input_X) shape: {shap_values[0].shape}")


# Aggregate SHAP values
# Assuming your task is sequence-based, we’ll average across time steps (axis=1) and instances (axis=0)
mean_shap = np.abs(shap_values[0]).mean(axis=(0, 1))

# Get feature names from model
feature_names = model.FEATURES  # assuming it's a list of strings

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
plt.savefig("shap_explanation_fig.png")


#### FORCE PLOT #####
# shap.initjs() # This is for interactive vizualizations


# Choose one sample (e.g., first sample and first time step)
sample_idx = 0
time_step = 0

# Get the SHAP values and input for that time step
shap_vals_single = shap_values[0][sample_idx, time_step]
input_single = input_x[sample_idx, time_step].detach().cpu().numpy()


def force_plot(base_value: int = 0):
    print("Force plot...")
    # Plot the force plot
    plot = shap.force_plot(
        base_value=0,  # If your model has no baseline, 0 is fine
        shap_values=shap_vals_single,
        features=input_single,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    # Save the force plot to a file (using matplotlib)
    plt.savefig("shap_force_plot.png", format="png", dpi=300, bbox_inches="tight")


#### SUMMARY PLOT #####
def summary_plot(cls: int = 0, idx: int = 0):

    print("Summary plot...")

    input_np = input_x.detach().cpu().numpy()
    input_x_single = input_np[
        idx, :, :
    ]  # Shape: (12, 13) - Take the first batch (12 time steps, 13 features)

    # Flatten for plotting
    input_x_single_flat = input_x_single.reshape(
        -1, input_x_single.shape[-1]
    )  # Shape: (12 * 13, 13)

    # Similarly, flatten SHAP values for the same batch
    shap_vals_single = shap_values[cls][
        :, idx, :
    ]  # Shape: (12, 13) - Take the first batch, all time steps
    shap_vals_single_flat = shap_vals_single.reshape(
        -1, shap_vals_single.shape[-1]
    )  # Shape: (12 * 13, 13)

    # Plot the summary plot
    shap.summary_plot(
        shap_vals_single_flat,  # Flattened SHAP values for the first batch
        input_x_single_flat,  # Flattened input features for the first batch
        feature_names=model.FEATURES,  # List of feature names
        show=False,
    )

    # Step 3: Save the plot
    # plt.tight_layout()
    plt.savefig("shap_summary_plot.png", format="png", dpi=300, bbox_inches="tight")


### WATEFALL PLOT #####
def waterfall_plot(idx: int = 0):
    print("Waterfall plot...")
    shap_values_for_instance = shap_vals_single

    input_np = input_x.detach().cpu().numpy()
    input_x_single = input_np[
        idx, :, :
    ]  # Shape: (12, 13) - Take the first batch (12 time steps, 13 features)

    # We need the input features for the instance
    input_features_for_instance = input_x_single[idx, :]

    print(input_x[idx : idx + 1].shape)

    output_idx = 0
    base_value = model.shap_predict(input_x[idx : idx + 1])[0, output_idx].item()

    # Create the waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_for_instance,  # SHAP values for the instance
            base_values=base_value,
            data=input_features_for_instance,  # The input features
            feature_names=model.FEATURES,  # Feature names
        )
    )

    # Save the waterfall plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9, top=0.9)
    plt.savefig("shap_waterfall_plot.png", format="png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    print("Creating plots...")
    force_plot()
    summary_plot()
    waterfall_plot()
    print("Done!")
