import shap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.save_model import get_model

from src.local_model.model import BaseLSTM
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.utils.log import setup_logging

# Setup logging
setup_logging()


# Get model
model: BaseLSTM = get_model(name="test_base_model.pkl")
# model.to(model.device)

# Get input data
loader = StatesDataLoader()

states_dict = loader.load_states(states=["Czechia"])

train_dict, test_dict = loader.split_data(
    states_dict=states_dict,
    sequence_len=model.hyperparameters.sequence_length,
    split_rate=0.8,
)

print(f"Sequence length: {model.hyperparameters.sequence_length}")

# Create sequences
input_sequences, _ = loader.create_train_sequences(
    states_data=train_dict,
    sequence_len=model.hyperparameters.sequence_length,
    features=model.FEATURES,
)

input_sequences.requires_grad = True


# Set training batches as input_x
model.train()
model.lstm.train()


model.to(device=model.device)
input_x = input_sequences.to(device=model.device)

print(f"INPUT X SHAPE = {input_x.shape}")

# Compute shap values
torch.backends.cudnn.enabled = False
# with torch.enable_grad():
explainer = shap.GradientExplainer(model, input_x)
# Compute SHAP values
shap_values = explainer.shap_values(input_x)
torch.backends.cudnn.enabled = True
print(f"SHAP_VALUES[0] FROM INPUT X SHAPE = {shap_values[0].shape}")


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
plt.savefig("test_explanation_fig.png")


## FORCE PLOT
# shap.initjs() # This is for interactive vizualizations

# Choose one sample (e.g., first sample and first time step)
sample_idx = 0
time_step = 0

# Get the SHAP values and input for that time step
shap_vals_single = shap_values[0][sample_idx, time_step]
input_single = input_x[sample_idx, time_step].detach().cpu().numpy()


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


# SUMMARY PLOT
# shap_vals = shap_values[0]  # shape: (batch, seq_len, features)
# cls = 0
# idx = 0


# shap_vals = shap_values[0]  # shape: (batch_size, sequence_len, input_features_num)

# shap_vals_avg = shap_vals.mean(axis=1)

# # Convert input data to numpy format (for SHAP plotting)


# # print(shap_values[cls][:, idx, :].shape)
# # print(input_x[:, idx, :].shape)
# shap.summary_plot(
#     # shap_values[cls][:, idx, :],
#     # input_x[idx, :, :],
#     shap_vals_avg,
#     input_np,
#     feature_names=model.FEATURES,
#     show=False,
# )  # Feature names (list of feature names))


# Assuming you want to work with the first batch
input_np = input_x.detach().cpu().numpy()
input_x_single = input_np[
    0, :, :
]  # Shape: (12, 13) - Take the first batch (12 time steps, 13 features)

# Flatten for plotting
input_x_single_flat = input_x_single.reshape(
    -1, input_x_single.shape[-1]
)  # Shape: (12 * 13, 13)

# Similarly, flatten SHAP values for the same batch
shap_vals_single = shap_values[0][
    :, 0, :
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

# Save the p
print("Here")

# Step 3: Save the plot
# plt.tight_layout()
plt.savefig("shap_summary_plot.png", format="png", dpi=300, bbox_inches="tight")
