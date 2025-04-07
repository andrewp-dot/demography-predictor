# Standard library imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import DatasetCreatorSettings

settings = DatasetCreatorSettings()

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
DATASET_PATH = os.path.join(
    settings.save_dataset_path,
    f"dataset_{settings.dataset_version}",
    f"dataset_{settings.dataset_version}.csv",
)
df = pd.read_csv(DATASET_PATH)

stateless_df = df.drop(columns=["country name"])

# Compute the correlation matrix
corr_matrix = stateless_df.corr()

# Display the correlation matrix
print(corr_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")

plt.savefig("feature_correlation_matrix.png")
# plt.show()

# Save figure
