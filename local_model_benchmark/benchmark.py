"""
In this file are experiments with local model.
"""

# Standard libraries
import logging
from typing import Dict, List


# Custom imports
from config import Config
from src.utils.log import setup_logging

from local_model_benchmark.experiments.data_experiments import (
    Experiment1,
    Experiment2,
    Experiment2_1,
    Experiment3,
    Experiment3_1,
)
from local_model_benchmark.experiments.model_experiments import (
    LSTMOptimalParameters,
    RNNvsStatisticalMethods,
)
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel

from local_model_benchmark.experiments.base_experiment import Experiment


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: doc comments


# List of available experimets
experiment_list: List[Experiment] = [
    Experiment1(),
    Experiment2(),
    Experiment2_1(),
    Experiment3(),
    Experiment3_1(),
    # Model experiments
    LSTMOptimalParameters(),
    RNNvsStatisticalMethods(),
]

# Setup experiments -> convert experiment list to dict

AVAILABLE_EXPERIMENTS: Dict[str, Experiment] = {
    exp.exp.name: exp for exp in experiment_list
}


def print_available_experiments(
    with_description: bool = False,
) -> Dict[str, Experiment]:

    print("Available experiments:")
    print("-" * 50)
    if with_description:
        for exp_name, exp in AVAILABLE_EXPERIMENTS.items():
            print(f" {exp_name}".ljust(50), exp.exp.description)

        return

    for exp_name in AVAILABLE_EXPERIMENTS.keys():
        print(f" {exp_name}")

    return


def run_experiments(
    experiments: List[str], state: str = "Czechia", split_rate: float = 0.8
) -> None:

    try:
        to_run_experiments_dict: Dict[str, Experiment] = {
            name: AVAILABLE_EXPERIMENTS[name] for name in experiments
        }
    except KeyError as e:
        logger.error(f"There is no experiment named {e}. Running no experiments.")
        return

    for name, exp in to_run_experiments_dict.items():
        logger.info(f"Running experiment {name} ...")
        exp.run(state=state, split_rate=split_rate)
        logger.info(f"Exp {name} Done!")


def run_all_experiments(state: str = "Czechia", split_rate: float = 0.8) -> None:
    for name, exp in AVAILABLE_EXPERIMENTS.items():
        logger.info(f"Running experiment {name} ...")
        exp.run(state=state, split_rate=split_rate)
        logger.info(f"Exp {name} Done!")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Print all experiments
    print_available_experiments(True)


# Use different scaler(s)
# 1. Robust scalers, MinMax, Logaritmic transformation

# Odstranit outliers?
# Zaokrúhlenie dát, čo s nimi?

# Spájanie modelov:
# Stacking? - priemerovanie vysledkov viacerych modelov subezne
# Boosting? - Ada boost (les neuroniek? :D) , XGBoost

# TODO: try different loss functions

#  criterion = nn.HuberLoss(delta=1.0)

# Interesting to try:
# Feature engineering -> correlations, stationary features, non-stationary features
# Maybe delete the COVID years in order to known it's influence on model accuracy?
