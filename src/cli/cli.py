# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import click
from typing import Literal, Optional, List


# Custom imports
from src.utils.log import setup_logging
from model_experiments.main import (
    print_available_experiments,
    run_all_experiments,
    run_experiments,
)

from src.train_scripts.train_predictors import (
    train_aging_predictor,
    train_gender_dist_predictor,
)

from src.shap_explainer.explain import explain_cli

from model_experiments.experiments.predictor_experiments import run_all


@click.group()
def cli():
    """
    Defines the main cli group
    """
    setup_logging()
    pass


@cli.group()
def model_experiments():
    """
    Defines the main cli group
    """
    setup_logging()
    pass


@cli.command()
@click.option(
    "--name",
    type=str,
    help="The name of the required (trained) pipeline to explain.",
    required=True,
)
@click.option(
    "--type",
    type=click.Choice(["feature", "target", "full-predictor"], case_sensitive=False),
    help="The type of the required (trained) pipeline to explain.",
    required=True,
)
@click.option(
    "--experimental",
    is_flag=True,
    help="If set to true, loads the pipeline from the trained_modes/experimental_models folder",
    default=False,
)
@click.option(
    "--core-metric",
    type=click.Choice(["mae", "mse", "rmse", "r2", "mape"]),
    help="All states will be sorted by this metric.",
    default="rmse",
)
@click.option(
    "--additional-states",
    type=str,
    help='List of additional states included in model explaination. List states as a comma separated strings string, e.g. --additional-states "Czechia, United States"',
    default=None,
)
def explain(
    name: str,
    type: Literal["feature", "target", "full-predictor"],
    experimental: bool,
    core_metric: Literal["mae", "mse", "rmse", "r2", "mape"],
    additional_states: str,
):

    # Parse states
    additional_states_list = (
        [state.strip() for state in additional_states.split(",")]
        if additional_states
        else []
    )

    explain_cli(
        pipeline_type=type,
        name=name,
        is_experimental=experimental,
        core_metric=core_metric,
        additional_states=additional_states_list,
    )


# Subgroup for the experiment group
# poetry run cli model_experiments --all
# poetry run cli model_experiments --experiments "name 1"  "name 2"
# poetry run cli model_experiments --list


@model_experiments.command()
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed descriptions of available experiments.",
)
def list(detailed):
    """
    Lists experiments. If detailed, lists experiments with their descriptions
    """

    print_available_experiments(with_description=detailed)


@model_experiments.command()
@click.option(
    "--all",
    is_flag=True,
    help="Runs all experiments with default parameters. Defualt parameters are defined in the config file.",
)
@click.option(
    "--state",
    type=str,
    help="Specifies the state to run experiment or evaluation on.",
    default="Czechia",
)
@click.option(
    "--split-rate",
    type=float,
    help="Size of the training set. For example 0.8 is equal to 80% of the dataset is used for training.",
    default=0.8,
)
@click.argument(
    "experiments",
    nargs=-1,
)
def run(all, experiments, state, split_rate):
    """
    Runs experiments based on given options.
    """
    setup_logging()

    if all:
        run_all_experiments(state=state, split_rate=split_rate)

    elif experiments:
        run_experiments(experiments=experiments, state=state, split_rate=split_rate)


@cli.group()
def train():
    """
    Defines the main cli group
    """
    setup_logging()
    pass


@train.command()
@click.option(
    "--name",
    type=str,
    help="Name of the predictor. It is also a model key.",
    required=True,
)
@click.option(
    "--type",
    type=click.Choice(["LSTM", "ARIMA"], case_sensitive=False),
    help="Type of the model. You can choose from options: LSTM, ARIMA",
    default="ARIMA",
)
@click.option(
    "--wealth-groups",
    type=str,
    help='List of comma separated wealth groups. as a comma separated strings string, e.g. --wealth-groups "high_income, lower_middle_income"',
)
@click.option(
    "--geolocation-groups",
    type=str,
    help='List of comma separated geolocation groups. as a comma separated strings string, e.g. --states "europe, north america"',
)
@click.option(
    "--states",
    type=str,
    help='List of included in training states as a comma separated strings string, e.g. --states "Czechia, United States"',
)
@click.option(
    "--modify-for-target-model",
    is_flag=True,
    help="If specified, modifed data will be used to train the target prediction model. By deafult, data are adjusted for feature target model",
)
def age_predictor(
    name: str,
    type: Literal["LSTM", "ARIMA"],
    wealth_groups: Optional[str],
    geolocation_groups: Optional[str],
    states: Optional[str],
    modify_for_target_model: bool,
):

    # Parse data arguments
    if states:
        states: List[str] = [s.strip() for s in states.split(",")]
    if wealth_groups:
        wealth_groups: List[str] = [s.strip() for s in wealth_groups.split(",")]
    if geolocation_groups:
        geolocation_groups: List[str] = [
            s.strip() for s in geolocation_groups.split(",")
        ]

    train_aging_predictor(
        name=name,
        model_type=type,
        states=states,
        wealth_groups=wealth_groups,
        geolocation_groups=geolocation_groups,
        modify_for_target_model=modify_for_target_model,
    )


@train.command()
@click.option(
    "--name",
    type=str,
    help="Name of the predictor. It is also a model key.",
    required=True,
)
@click.option(
    "--type",
    type=click.Choice(["LSTM", "ARIMA"], case_sensitive=False),
    help="Type of the model. You can choose from options: LSTM, ARIMA",
    default="ARIMA",
)
@click.option(
    "--wealth-groups",
    type=str,
    help='List of comma separated wealth groups. as a comma separated strings string, e.g. --wealth-groups "high_income, lower_middle_income"',
)
@click.option(
    "--geolocation-groups",
    type=str,
    help='List of comma separated geolocation groups. as a comma separated strings string, e.g. --states "europe, north america"',
)
@click.option(
    "--states",
    type=str,
    help='List of included in training states as a comma separated strings string, e.g. --states "Czechia, United States"',
)
@click.option(
    "--modify-for-target-model",
    is_flag=True,
    help="If specified, modifed data will be used to train the target prediction model. By deafult, data are adjusted for feature target model",
)
def dist_predictor(
    name: str,
    type: Literal["LSTM", "ARIMA"],
    wealth_groups: Optional[str],
    geolocation_groups: Optional[str],
    states: Optional[str],
    modify_for_target_model: bool,
):

    # Parse data arguments
    if states:
        states: List[str] = [s.strip() for s in states.split(",")]
    if wealth_groups:
        wealth_groups: List[str] = [s.strip() for s in wealth_groups.split(",")]
    if geolocation_groups:
        geolocation_groups: List[str] = [
            s.strip() for s in geolocation_groups.split(",")
        ]

    train_gender_dist_predictor(
        name=name,
        model_type=type,
        states=states,
        wealth_groups=wealth_groups,
        geolocation_groups=geolocation_groups,
        modify_for_target_model=modify_for_target_model,
    )


@cli.group()
def predictor_experiments():
    """
    Defines the main cli group
    """
    setup_logging()
    pass


@predictor_experiments.command()
@click.option(
    "--name",
    type=str,
    help="Name of the predictor to run predictor experiments with.",
    required=True,
)
def run(name: str):
    run_all(pipeline_name=name)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run cli
    cli()
