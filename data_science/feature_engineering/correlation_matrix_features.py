# Standard library imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import List, Tuple
import numpy as np

import pprint


# Custom imports
from config import DatasetCreatorSettings

settings = DatasetCreatorSettings()


# Load the dataset (replace 'your_dataset.csv' with the actual file path)


def create_correlation_matrix(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    stateless_df = df.drop(columns=["country name"])

    # Compute the correlation matrix
    corr_matrix = stateless_df.corr()

    return corr_matrix


def display_corr_matrix(
    corr_matrix: pd.DataFrame, save_path: str | None = None, only_triangle: bool = False
) -> Figure:

    # Create a mask for the upper triangle
    mask = None
    if only_triangle:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot the heatmap
    corr_matrix_fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Matrix Heatmap")

    if save_path:
        plt.savefig(save_path)

    # Return the correlation matrix figure
    return corr_matrix_fig


def get_highly_correlated_features(
    corr_matrix: pd.DataFrame, threshold: float = 0.8
) -> List[Tuple[str, str]]:
    """
    Get a list of highly correlated features from the correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix.
        threshold (float): The correlation threshold to consider.

    Returns:
        list: A list of tuples containing pairs of highly correlated features.
    """
    correlated_pairs: List[Tuple[str, str]] = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_pairs.append((colname, corr_matrix.columns[j]))

    return correlated_pairs


def main():
    DATASET_PATH = os.path.join(
        settings.save_dataset_path,
        f"dataset_{settings.dataset_version}",
        f"dataset_{settings.dataset_version}.csv",
    )

    corr_matrix = create_correlation_matrix(DATASET_PATH)

    print(corr_matrix)

    display_corr_matrix(
        corr_matrix,
        save_path=os.path.join(
            settings.visualizations_dir, "feature_correlation_matrix.png"
        ),
        only_triangle=True,
    )
    plt.show()

    higly_correlated_features = get_highly_correlated_features(
        corr_matrix=corr_matrix, threshold=0.9
    )

    pprint.pprint(higly_correlated_features)


if __name__ == "__main__":
    main()
