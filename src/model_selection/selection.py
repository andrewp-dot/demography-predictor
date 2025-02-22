# Standard library imports
import logging
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Pycaret imports
from pycaret.regression import *

# Custom imports
from config import setup_logging, Config


# Get config
config = Config()

# Get logger
logger = logging.getLogger("method_selection")


# TODO:
# 1. Load data
def load_data() -> pd.DataFrame:
    # Load data from file
    data = pd.read_csv(config.dataset_path)

    # Rename columns to all lower
    mapper = {col: col.lower() for col in data.columns}

    data.rename(columns=mapper, inplace=True)
    return data


# 2. Create regression experiment
def regression_experiment(
    data: pd.DataFrame, target: str, session_id: int
) -> List[Any]:
    # Setup experiment
    experiment = setup(data, target=target, session_id=session_id)

    best_model = compare_models()

    metrics_table = pull()  # Pulls the last displayed table

    # Plot table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=metrics_table.values,
        colLabels=metrics_table.columns,
        cellLoc="center",
        loc="center",
    )

    plt.title("Model Performance Comparison", fontsize=14)
    plt.show()

    return best_model


# 3. plot the results
def plot_results(best_model: List[Any]) -> None:
    # Plot Residuals
    plot_model(best_model, plot="residuals")

    # Plot Prediction Error
    plot_model(best_model, plot="error")

    # Plot Feature Importance
    plot_model(best_model, plot="feature")

    # Plot Learning Curve
    plot_model(best_model, plot="learning")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # 1. Load data
    data_df = load_data()

    logger.debug(f"Data columns:\n {data_df.columns}")

    # 2. Run regression experiment
    best_model = regression_experiment(data=data_df, target="population, total")

    # 3. Plot model
    # plot_results(best_model=best_model)
