"""
In this file are experiments with local model.
"""

# Standard libraries
import logging
from typing import Dict, List
import traceback
import pprint


# Custom imports
from config import Config
from src.utils.log import setup_logging

from local_model_benchmark.experiments.data_experiments import (
    DataUsedForTraining,
    StatesByGroup,
)
from local_model_benchmark.experiments.model_experiments import (
    FeaturePredictionSeparatelyVSAtOnce,
    FineTunedModels,
    CompareWithStatisticalModels,
)

# from local_model_benchmark.experiments.new_data_experiments import DataUsedForTraining
# from local_model_benchmark.experiments.new_model_experiments import

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import RNNHyperparameters, BaseLSTM, EvaluateModel

from local_model_benchmark.experiments.base_experiment import Experiment


settings = Config()
logger = logging.getLogger("benchmark")

# TODO: Define experiments
# TODO: doc comments

# Setup logging if it is not
# setup_logging()

data_experiments: List[Experiment] = [
    # s
]

# List of available experimets
model_experiments: List[Experiment] = [
    # LSTMOptimalParameters(),
    # RNNvsStatisticalMethodsSingleFeature(),
    # RNNvsStatisticalMethods(),
    # FineTuneLSTMExp(),
]


# Setup experiments -> convert experiment list to dict

AVAILABLE_EXPERIMENTS_BY_GROUP: Dict[str, Dict[str, Experiment]] = {
    "model_experiments": {exp.exp.name: exp for exp in data_experiments},
    "data_experiments": {exp.exp.name: exp for exp in model_experiments},
}

AVAILABLE_EXPERIMENTS: Dict[str, Experiment] = {
    exp.exp.name: exp for exp in (model_experiments + data_experiments)
}


def print_available_experiments(
    with_description: bool = False,
) -> Dict[str, Experiment]:

    print("Available experiments:")
    print("-" * 50)

    for exp_group, experiments_dict in AVAILABLE_EXPERIMENTS_BY_GROUP.items():
        # Print experiments by group
        print(f"\n{exp_group}:\n")

        # Print single experiments
        if with_description:
            for exp_name, exp in experiments_dict.items():
                print(f" {exp_name}".ljust(50), exp.exp.description)

        else:
            for exp_name in experiments_dict.keys():
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

    experiments_failed: List[str] = []
    for name, exp in AVAILABLE_EXPERIMENTS.items():
        logger.info(f"Running experiment {name} ...")

        try:
            exp.run(state=state, split_rate=split_rate)
        except Exception as e:
            # Add experiments to failed experiments
            experiments_failed.append(exp.exp.name)

            # Print error message with traceback
            logger.error(f"Exp {name} failed! Reason:")
            traceback.print_exc()
            continue

        logger.info(f"Exp {name} Done!")

    # If there are any, print the names of failed experiments
    if experiments_failed:
        formatted_failed_experiments = pprint.pformat(experiments_failed)
        logger.error(f"Failed experiemnts: {formatted_failed_experiments}")


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
