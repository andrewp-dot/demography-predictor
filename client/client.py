"""
This is just testing client for the API to access deployed model. Can be also used as a cookbook.
"""

# Standard library imports
import requests
import argparse
from typing import Dict, List
import pprint
import pandas as pd

# Custom imports
from config import Config
from src.preprocessors.state_preprocessing import StateDataLoader
from src.api.models import (
    PredictionRequest,
)  # This is used just for easier implementation, can use Dict or implement custom pydantic class instead

# Get settings
settings = Config()

BASE_URL: str = f"{settings.api_host}:{settings.api_port}"
AVAILABLE_ENDPOINTS: Dict[str, str] = {"info": "/info", "predict": "/predict"}

# TODO:
# Implement this robust for info gathering and endpoint choosing -> maybe cli?


def send_base_prediction_request(
    state: str, model_key: str, target_year: int
) -> requests.Response:
    """
    Send

    Args:
        state (str): _description_

    Returns:
        requests.Response: _description_
    """

    # Get headers
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_ACCESS_TOKEN",  # If authentication is needed
    }

    # Prepare data
    state_loader = StateDataLoader(state=state)
    state_data_df = state_loader.load_data()

    # Convert dataframe to list of dict, where single dict represents one row
    input_data = state_data_df.to_dict(orient="records")

    request = PredictionRequest(
        model_key=model_key, state=state, input_data=input_data, target_year=target_year
    )

    response = requests.post(
        url=f"http://{BASE_URL}{AVAILABLE_ENDPOINTS['predict']}",
        headers=headers,
        json=request.model_dump(),
    )

    return response


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--target-year", type=int)
    parser.add_argument("--model-key", type=str)
    parser.add_argument("--state", type=str)

    # Excract arguments
    args = parser.parse_args()
    target_year = args.target_year
    model_key = args.model_key
    state = args.state.capitalize()

    # Send the base prediction request
    response = send_base_prediction_request(
        state=state, model_key=model_key, target_year=target_year
    )

    prediction_df = pd.DataFrame(response.json()["predictions"])
    print(prediction_df)


if __name__ == "__main__":
    main()
