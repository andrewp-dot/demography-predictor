import shap
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.save_model import get_model

from src.local_model.model import BaseLSTM
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


# Force plot !!

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

# Create sequences
input_sequences, target_sequences = loader.create_train_sequences(
    states_data=train_dict,
    sequence_len=model.hyperparameters.sequence_length,
    features=model.FEATURES,
)

input_sequences.requires_grad = True


# Create batches
# training_batches, _ = loader.create_train_batches(
#     input_sequences=input_sequences, target_sequences=target_sequences, batch_size=1
# )

# training_batches, _, _ = loader.preprocess_train_data_batches(
#     states_train_data_dict=train_dict,
#     hyperparameters=model.hyperparameters,
#     features=model.FEATURES,
#     scaler=model.SCALER,
# )

# Required for SHAP gradient-based methods
# training_batches.requires_grad = True

# Set training batches as input_x


model.train()
model.lstm.train()


model.to(device=model.device)
input_x = input_sequences.to(device=model.device)

torch.backends.cudnn.enabled = False
with torch.enable_grad():
    explainer = shap.GradientExplainer(model, input_x)
    # Compute SHAP values
    shap_values = explainer.shap_values(input_x)
torch.backends.cudnn.enabled = True


# Aggregate SHAP values
# Assuming your task is sequence-based, we’ll average across time steps (axis=1) and instances (axis=0)
mean_shap = np.abs(shap_values[0]).mean(axis=(0, 1))

# Get feature names from model
feature_names = model.FEATURES  # assuming it's a list of strings

# Plot
plt.figure(figsize=(10, 5))
plt.bar(feature_names, mean_shap)
plt.xlabel("Features")
plt.ylabel("Mean |SHAP Value| (Feature Importance)")
plt.title("Overall Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("test_explanation_fig.png")


shap.initjs()

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
)

# Save the force plot to a file (using matplotlib)
plt.savefig("shap_force_plot.png", format="png", dpi=300, bbox_inches="tight")
