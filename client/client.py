"""
This is just testing client for the API to access deployed model. Can be also used as a cookbook.
"""

# Standard library imports
import click
from typing import Dict, List
import pandas as pd

# Custom imports
from config import Config
from client.client_requests import send_info_request, send_base_prediction_request

# Get settings
settings = Config()


@click.group()
def client():
    """Creates client command group"""
    pass


@client.command()
def info():
    response = send_info_request()
    print(response.json())


@client.command()
@click.option(
    "--state",
    type=str,
    help="The data of the 'state' will be used for prediction.",
    required=True,
)
@click.option(
    "--model-key",
    type=str,
    help="Predict the data by specified model key.",
    required=True,
)
@click.option(
    "--target-year",
    type=int,
    help="The year to the target data will be predicted.",
    required=True,
)
def predict(state: str, model_key: str, target_year: int):
    # Send the base prediction request
    response = send_base_prediction_request(
        state=state.capitalize(), model_key=model_key, target_year=target_year
    )

    if response.status_code == 200:
        prediction_df = pd.DataFrame(response.json()["predictions"])
        print(prediction_df)
    else:
        print(response.text)


@client.command()
@click.option(
    "--state",
    type=str,
    help="The data of the 'state' will be used for prediction.",
    required=True,
)
@click.option(
    "--model-key",
    type=str,
    help="Predict the data by specified model key.",
    required=True,
)
@click.option(
    "--target-year",
    type=int,
    help="The year to the target data will be predicted.",
    required=True,
)
def lakmoos_predict(state: str, model_key: str, target_year: int):
    # Send the lakmoos prediction request for getting distribution curve in the target year
    response = send_base_prediction_request(
        state=state.capitalize(),
        model_key=model_key,
        target_year=target_year,
        lakmoos_predict=True,
    )

    if response.status_code == 200:
        try:
            # Get and print predictions
            prediction_df = pd.DataFrame(response.json()["predictions"])
            print(prediction_df)

            # Get and print distribution for the predictions
            distribution_df = pd.DataFrame(response.json()["distribution"])
            print(distribution_df)
        except KeyError as e:
            print(str(e))

    else:
        print(response.text)


if __name__ == "__main__":
    client()
