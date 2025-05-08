# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard libraries
import logging
from typing import Dict, List, Optional, Literal
import traceback
import pprint


# Custom imports
from config import Config
from src.utils.log import setup_logging

from model_experiments.experiments.data_experiments import (
    DataUsedForTraining,
    StatesByGroup,
)
from model_experiments.experiments.model_experiments import (
    FineTunedModels,
    CompareWithStatisticalModels,
)

# Selection model experiments
from model_experiments.model_selection.feature_model_selection import (
    FeatureModelExperiment,
)
from model_experiments.model_selection.target_model_selection import (
    TargetModelSelection,
)
from model_experiments.base_experiment import BaseExperiment


settings = Config()
logger = logging.getLogger("model_experiments")


# TODO: rework this to run it by CLI


data_experiments: List[BaseExperiment] = [
    # s
]

# List of available experimets
model_experiments: List[BaseExperiment] = [
    # LSTMOptimalParameters(),
    # RNNvsStatisticalMethodsSingleFeature(),
    # RNNvsStatisticalMethods(),
    # FineTuneLSTMExp(),
]


# Setup experiments -> convert experiment list to dict


AVAILABLE_EXPERIMENTS: Dict[str, BaseExperiment] = {
    exp.name: exp for exp in (model_experiments + data_experiments)
}


# Model selection experiments
MODEL_SELECTION_EXPERIMENTS: Dict[str, BaseExperiment] = {
    # Create experiment
    "feature_model": FeatureModelExperiment(
        description="Compare models for predicting all features which are used for target predictions."
    ),
    "target_model_aging": TargetModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data.",
        target_group_prefix="aging",
    ),
    "target_model_pop_total": TargetModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data.",
        target_group_prefix="pop_total",
    ),
    "target_model_gender_dist": TargetModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data.",
        target_group_prefix="gender_dist",
    ),
}


# All experiments by group
AVAILABLE_EXPERIMENTS_BY_GROUP: Dict[str, Dict[str, BaseExperiment]] = {
    "model_experiments": {exp.name: exp for exp in data_experiments},
    "data_experiments": {exp.name: exp for exp in model_experiments},
    "model_selection": MODEL_SELECTION_EXPERIMENTS,
}


def print_available_experiments(
    with_description: bool = False,
) -> Dict[str, BaseExperiment]:

    print("Available experiments:")
    print("-" * 50)

    # 1. print model selection experiments

    for exp_group, experiments_dict in AVAILABLE_EXPERIMENTS_BY_GROUP.items():
        # Print experiments by group
        print(f"\n{exp_group}:\n")

        # Print single experiments
        if with_description:
            for exp_name, exp in experiments_dict.items():
                print(f" {exp_name}".ljust(50), exp.description)

        else:
            for exp_name in experiments_dict.keys():
                print(f" {exp_name}")

    return


# Fix this
def run_experiments(
    experiments: List[str], state: str = "Czechia", split_rate: float = 0.8
) -> None:

    try:
        to_run_experiments_dict: Dict[str, BaseExperiment] = {
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


## Model selection experiments
def run_feature_model_selection(
    split_rate: float = 0.8,
    force_retrain: bool = False,
    only_rnn_retrain: bool = False,
    evaluation_states: Optional[List[str]] = None,
) -> None:
    # Create experiment
    feature_model_selection = AVAILABLE_EXPERIMENTS_BY_GROUP["model_selection"][
        "feature_model"
    ]

    # Run the experiment
    feature_model_selection.run(
        split_rate=split_rate,
        force_retrain=force_retrain,
        only_rnn_retrain=only_rnn_retrain,
        evaluation_states=evaluation_states,
    )


def run_target_model_selection(
    split_rate: float = 0.8,
    force_retrain: bool = False,
    only_rnn_retrain: bool = False,
    evaluation_states: Optional[List[str]] = None,
    exp_type: Literal["aging", "pop_total", "gender_dist"] = "aging",
) -> None:

    target_model_selection = AVAILABLE_EXPERIMENTS_BY_GROUP["model_selection"][
        f"target_model_{exp_type}"
    ]

    target_model_selection.run(
        split_rate=split_rate,
        force_retrain=force_retrain,
        only_rnn_retrain=only_rnn_retrain,
        evaluation_states=evaluation_states,
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Print all experiments
    print_available_experiments(True)
