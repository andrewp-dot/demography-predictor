# Standard library imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import List, Tuple, Dict, Literal
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
    features_to_target_correlation: bool = False,
) -> pd.DataFrame:
    """
    Create a correlation matrix from the dataset.

    Args:
        dataset_path (str): Path to the existing dataset.
        exclude_targets (bool): Whether to exclude target features. Defaults to True.
        exclude_highly_correlated (bool): Excludes columns targeted as hihgly correlated.

    Returns:
        out: pd.DataFrame: Correlation matrix datframe.
    """
    df = pd.read_csv(dataset_path)

    df = df.drop(
        columns=["country_name"]
    )  # remove this because it is not numerical feature

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

    # Compute the correlation matrix
    corr_matrix = df.corr()

    if features_to_target_correlation:
        corr_matrix = corr_matrix.loc[
            feature_df.columns, settings.ALL_POSSIBLE_TARGET_FEATURES
        ]

    return corr_matrix


def create_feature_target_correlation_matrix(
    dataset_path: str, exclude_highly_correlated: bool = False
) -> pd.DataFrame:
    # Then select only the part you want (features vs targets)
    df = pd.read_csv(dataset_path)

    df = df.drop(
        columns=["country_name"]
    )  # remove this because it is not numerical feature

    # Drop columns with high correlation
    if exclude_highly_correlated:
        df = df.drop(columns=HIGHLY_CORRELATED_COLUMNS)

    # Exclude target features
    feature_df = df.drop(columns=settings.ALL_POSSIBLE_TARGET_FEATURES)

    # Put the targets to the end
    df = pd.concat([feature_df, df[settings.ALL_POSSIBLE_TARGET_FEATURES]], axis=1)

    # Compute the correlation matrix
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.loc[
        feature_df.columns, settings.ALL_POSSIBLE_TARGET_FEATURES
    ]

    return corr_matrix


def display_corr_matrix(
    corr_matrix: pd.DataFrame,
    save_path: str | None = None,
    only_triangle: bool = False,
    language: Literal["sk", "en"] = "en",
) -> Figure:

    # Create a mask for the upper triangle
    mask = None
    if only_triangle:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot the heatmap
    number_of_columns = max(len(corr_matrix.columns), len(corr_matrix))
    feature_row_size = 1.0
    corr_matrix_fig = plt.figure(
        figsize=(
            number_of_columns * feature_row_size,
            number_of_columns * feature_row_size,
        )
    )

    # Adjust font size
    font_scale = 1.0
    annot_fontsize = 12
    label_fontsize = 14

    sns.set_theme(font_scale=font_scale, context="paper")

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        annot_kws={"size": annot_fontsize},
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
    )

    TITLE: Dict[str, str] = {
        "en": "Correlation Matrix Heatmap",
        "sk": "Teplotná mapa korelačnej matice",
    }

    plt.title(TITLE[language], fontsize=label_fontsize + 2)
    plt.xticks(
        rotation=45, ha="right", fontsize=label_fontsize
    )  # <-- set label font size
    plt.yticks(fontsize=label_fontsize)  # <-- set y-axis label font size
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


def get_corr_matrix(
    plot_name: str = "all_feature_correlation_matrix.png",
    show_plot: bool = False,
    exclude_targets: bool = True,
    exclude_highly_correlated: bool = True,
    language: Literal["sk", "en"] = "en",
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
        language=language,
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


def get_feature_target_correlation(
    plot_name: str,
    exclude_highly_correlated: bool = False,
    language: Literal["sk", "en"] = "en",
) -> None:
    DATASET_PATH = os.path.join(
        settings.save_dataset_path,
        f"dataset_{settings.dataset_version}",
        f"dataset_{settings.dataset_version}.csv",
    )

    corr_matrix = create_feature_target_correlation_matrix(
        DATASET_PATH,
        exclude_highly_correlated=exclude_highly_correlated,
    )

    print(corr_matrix)

    display_corr_matrix(
        corr_matrix,
        save_path=os.path.join(settings.visualizations_dir, plot_name),
        language=language,
    )


if __name__ == "__main__":

    # All correlation features including targetes
    get_corr_matrix(
        plot_name="only_feature_correlation_matrix.png",
        show_plot=True,
        exclude_targets=True,
        exclude_highly_correlated=False,
        language="sk",
    )

    get_corr_matrix(
        plot_name="only_feature_low_correlation_matrix.png",
        show_plot=True,
        exclude_targets=True,
        exclude_highly_correlated=True,
        language="sk",
    )

    # All features excluding high correlation features
    get_corr_matrix(
        plot_name="all_feature_low_correlation_matrix.png",
        show_plot=True,
        exclude_targets=False,
        exclude_highly_correlated=True,
        language="sk",
    )

    # All correlation features including targetes
    get_corr_matrix(
        plot_name="all_feature_correlation_matrix.png",
        show_plot=True,
        exclude_targets=False,
        exclude_highly_correlated=False,
        language="sk",
    )

    get_feature_target_correlation(
        plot_name="low_correlation_feature_target.png",
        exclude_highly_correlated=True,
        language="sk",
    )
