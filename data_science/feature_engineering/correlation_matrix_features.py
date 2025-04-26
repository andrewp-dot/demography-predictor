# Standard library imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import List, Tuple, Dict
import numpy as np

import pprint


# Custom imports
from config import DatasetCreatorSettings

settings = DatasetCreatorSettings()


# Load the dataset (replace 'your_dataset.csv' with the actual file path)
HIGHLY_CORRELATED_COLUMNS: List[str] = [
    "life expectancy at birth, total",
    "age dependency ratio",
    "rural population",
    "birth rate, crude",
    "adolescent fertility rate",
]


def create_correlation_matrix(
    dataset_path: str,
    exclude_targets: bool = True,
    exclude_highly_correlated: bool = True,
) -> pd.DataFrame:
    """
    Create a correlation matrix from the dataset.

    Args:
        dataset_path (str): Path to the existing dataset.
        exclude_targets (bool): Whether to exclude target features. Defaults to True.

    Returns:
        out: pd.DataFrame: Correlation matrix datframe.
    """
    df = pd.read_csv(dataset_path)

    # Exclude target features
    feature_df = df.drop(columns=settings.ALL_POSSIBLE_TARGET_FEATURES)

    if exclude_targets:

        # Get only features in here
        df = feature_df
    else:
        # Put the targets to the end
        df = pd.concat([feature_df, df[settings.ALL_POSSIBLE_TARGET_FEATURES]], axis=1)

    # Drop columns with high correlation
    if exclude_highly_correlated:
        df = df.drop(columns=HIGHLY_CORRELATED_COLUMNS)

    df = df.drop(
        columns=["country name"]
    )  # remove this because it is not numerical feature

    # Compute the correlation matrix
    corr_matrix = df.corr()

    return corr_matrix


def display_corr_matrix(
    corr_matrix: pd.DataFrame,
    save_path: str | None = None,
    only_triangle: bool = False,
) -> Figure:

    # Create a mask for the upper triangle
    mask = None
    if only_triangle:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot the heatmap
    number_of_columns = len(corr_matrix.columns)
    feature_row_size = 1.0
    corr_matrix_fig = plt.figure(
        figsize=(
            number_of_columns * feature_row_size,
            number_of_columns * feature_row_size,
        )
    )

    # Dynamically adjust font size based on number of columns -> bigger the number, bigger the font due to readableness
    if number_of_columns < 10:
        font_scale = 1.0
    elif number_of_columns < 20:
        font_scale = 1.4
    elif number_of_columns < 40:
        font_scale = 1.6
    else:
        font_scale = 1.8

    sns.set_theme(font_scale=font_scale, context="paper")

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
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    # Return the correlation matrix figure
    return corr_matrix_fig


def get_highly_correlated_features(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
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


def convert_highly_correlated_features_to_dict(
    highly_correlated_features: List[Tuple[str, str]],
) -> Dict[str, List[str]]:
    """
    Convert a list of highly correlated features into a dictionary.

    Args:
        highly_correlated_features (list): A list of tuples containing pairs of highly correlated features.

    Returns:
        dict: A dictionary where the keys are feature names and the values are lists of correlated features.
    """
    highly_correlated_dict: Dict[str, List[str]] = {}

    # For each feature find hihgly correlated features
    for feature1, feature2 in highly_correlated_features:

        # Create a dictionary entry for feature1 if it doesn't exist
        if feature1 not in highly_correlated_dict:
            highly_correlated_dict[feature1] = []

        # Append corrlated feature2 to the list of feature1
        highly_correlated_dict[feature1].append(feature2)

    # Return transformed dictionaryy
    return highly_correlated_dict


def main(
    plot_name: str = "all_feature_correlation_matrix.png",
    show_plot: bool = False,
    exclude_targets: bool = True,
    exclude_highly_correlated: bool = True,
) -> None:
    DATASET_PATH = os.path.join(
        settings.save_dataset_path,
        f"dataset_{settings.dataset_version}",
        f"dataset_{settings.dataset_version}.csv",
    )

    corr_matrix = create_correlation_matrix(
        DATASET_PATH,
        exclude_targets=exclude_targets,
        exclude_highly_correlated=exclude_highly_correlated,
    )

    print(corr_matrix)

    display_corr_matrix(
        corr_matrix,
        save_path=os.path.join(settings.visualizations_dir, plot_name),
        only_triangle=True,
    )

    higly_correlated_features = get_highly_correlated_features(
        corr_matrix=corr_matrix, threshold=0.8
    )

    higly_correlated_features_dict = convert_highly_correlated_features_to_dict(
        higly_correlated_features
    )

    print("-" * 100)
    print("Highly correlated feature pairs:")
    pprint.pprint(higly_correlated_features)

    print("-" * 100)

    print("Highly correlated features dictionary:")
    pprint.pprint(higly_correlated_features_dict)

    if show_plot:
        plt.show()


if __name__ == "__main__":

    # All correlation features including targetes
    main(
        plot_name="only_feature_correlation_matrix.png",
        show_plot=True,
        exclude_targets=True,
        exclude_highly_correlated=False,
    )

    main(
        plot_name="only_feature_low_correlation_matrix.png",
        show_plot=True,
        exclude_targets=True,
        exclude_highly_correlated=True,
    )

    # All features excluding high correlation features
    main(
        plot_name="all_feature_low_correlation_matrix.png",
        show_plot=True,
        exclude_targets=False,
        exclude_highly_correlated=True,
    )

    # All correlation features including targetes
    main(
        plot_name="all_feature_correlation_matrix.png",
        show_plot=True,
        exclude_targets=False,
        exclude_highly_correlated=False,
    )
