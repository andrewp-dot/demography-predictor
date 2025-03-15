import click

# TODO: define multigroup cli


@click.group()
def cli():
    """
    Defines the main cli group
    """
    pass


# Subgroup for the experiment group
# poetry run cli lbenchmark --all
# poetry run cli lbenchmark --experiments "name 1"  "name 2"
# poetry run cli lbenchmark --list
@cli.command()
@click.option(
    "--all",
    is_flag=True,
    help="Runs all experiments with default parameters. Defualt parameters are defined in the config file.",
)
@click.option(
    "--experiments",
    multiple=True,
    help="Runs the specified experiments. The experiments should be defined in the config file.",
)
@click.option(
    "--list",
    is_flag=True,
    help="List all the experiments defined in the config file.",
)
def lbenchmark(all, experiments, list):
    """
    Runs experiments based on given options.
    """

    if list:
        click.echo("Listing all experiments...")
        # Call the function to list experiments
    elif all:
        click.echo("Running all experiments with default parameters...")
        # Implement logic to run all experiments
    elif experiments:
        click.echo(f"Running specified experiments: {', '.join(experiments)}")
        # Implement logic to run specified experiments
    else:
        click.echo("No option provided. Use --help for usage information.")


if __name__ == "__main__":
    cli()
