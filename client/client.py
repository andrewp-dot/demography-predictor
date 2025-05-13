# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License


"""
This is just testing client for the API to access deployed model. Can be also used as a cookbook.
"""

# Standard library imports
import click
import pandas as pd
import matplotlib.pyplot as plt
import json

# Custom imports
from config import Config
from client.client_requests import (
    send_info_request,
    send_base_prediction_request,
    send_lakmoos_prediction_request,
)

# Get settings
settings = Config()


def plot_age_distribution_curve(distribution_df: pd.DataFrame) -> None:

    if "age" in distribution_df:
        plt.figure(figsize=(10, 4))
        plt.plot(distribution_df["age"], distribution_df["probability"])
        plt.show()


def plot_predictions(df: pd.DataFrame) -> None:

    try:
        prediction_years = df["year"]

        prediction_columns = [col for col in df.columns if "year" != col]

        # Create figure
        fig, axes = plt.subplots(
            len(prediction_columns),
            1,
            figsize=(10, 4 * len(prediction_columns)),
            sharex=True,
        )

        if len(prediction_columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, prediction_columns):
            ax.plot(prediction_years, df[col], marker="o", label=col)
            ax.set_ylabel(col)
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Year")
        fig.suptitle("Model Predictions Over Time", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except KeyError as e:
        print(f"[Plotting error]: {str(e)}")


@click.group()
def client():
    """Creates client command group"""
    pass


@client.command()
def info():
    response = send_info_request()
    try:
        print(response.json())
    except:
        print(response)


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
@click.option(
    "--show-plots",
    is_flag=True,
    help="If specified, plots the predictions.",
    required=False,
    default=False,
    show_default=True,
)
@click.option(
    "--save-response",
    is_flag=True,
    help="If specified, saves the response to a file.",
    required=False,
    default=False,
    show_default=True,
)
def predict(
    state: str,
    model_key: str,
    target_year: int,
    show_plots: bool,
    save_response: bool = False,
):
    # Send the base prediction request
    response = send_base_prediction_request(
        state=state, model_key=model_key, target_year=target_year
    )

    if response.status_code == 200:

        if save_response:
            # Save the response to a file
            with open("predict_response.json", "w") as f:
                json.dump(response.json(), f, indent=4)
            print("Response saved to predict_response.json")

        json_response = response.json()
        prediction_df = pd.DataFrame(json_response["predictions"])
        print(prediction_df)

        if show_plots:
            plot_predictions(df=prediction_df)
    else:
        print(f"{response.status_code}: {response.text}")


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
@click.option(
    "--max-age",
    type=int,
    help="The estimated max age of human. Get the probability distribution to this age.",
    required=False,  # Set this to optional
    default=100,
    show_default=True,  # Show default in help text
)
@click.option(
    "--show-plots",
    is_flag=True,
    help="If specified, plots the predictions.",
    required=False,
    default=False,
    show_default=True,
)
@click.option(
    "--save-response",
    is_flag=True,
    help="If specified, saves the response to a file.",
    required=False,
    default=False,
    show_default=True,
)
def lakmoos_predict(
    state: str,
    model_key: str,
    target_year: int,
    max_age: int,
    show_plots: bool,
    save_response: bool = False,
):
    """
    Prediction for lakmoos prediction endpoint. Gets the predictions and also the distribution of the required parameter.

    Args:
        state (str): Name of the state to predict.
        model_key (str): Name of the model used for prediction.
        target_year (int): The year of the last prediction.
    """
    # Send the lakmoos prediction request for getting distribution curve in the target year
    response = send_lakmoos_prediction_request(
        state=state,
        model_key=model_key,
        target_year=target_year,
        max_age=max_age,
    )

    if response.status_code == 200:
        try:
            # Get and print predictions
            if save_response:
                # Save the response to a file
                with open("lakmoos_predict_response.json", "w") as f:
                    json.dump(response.json(), f, indent=4)

                print("Response saved to lakmoos_predict_response.json")

            json_response = response.json()
            prediction_df = pd.DataFrame(json_response["predictions"])
            print(prediction_df)

            # Get and print distribution for the predictions
            distribution_df = pd.DataFrame(json_response["distribution"])
            print(distribution_df)

            if show_plots:
                plot_predictions(df=prediction_df)
                plot_age_distribution_curve(distribution_df=distribution_df)

        except KeyError as e:
            print(str(e))

    else:
        print(f"{response.status_code}: {response.text}")


if __name__ == "__main__":
    client()
