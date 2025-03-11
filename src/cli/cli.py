import click

# TODO: define multigroup cli


@click.group()
def cli():
    """
    Defines the main cli group
    """
    pass


@click.group()
def benchmark():
    """
    Benchmarking cli group
    """
    pass


if __name__ == "__main__":
    cli()
