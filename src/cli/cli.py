import click

from src.utils.log import setup_logging
from local_model_benchmark.benchmark import (
    print_available_experiments,
    run_all_experiments,
    run_experiments,
)


@click.group()
def cli():
    """
    Defines the main cli group
    """
    setup_logging()
    pass


@cli.group()
def lbenchmark():
    """
    Defines the main cli group
    """
    pass


# Subgroup for the experiment group
# poetry run cli lbenchmark --all
# poetry run cli lbenchmark --experiments "name 1"  "name 2"
# poetry run cli lbenchmark --list


@lbenchmark.command()
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


@lbenchmark.command()
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

    if all:
        run_all_experiments(state=state, split_rate=split_rate)

    elif experiments:
        run_experiments(experiments=experiments, state=state, split_rate=split_rate)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run cli
    cli()
