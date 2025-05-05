# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License


# Standard library imports
import requests
from typing import Dict

# Custom imports
from config import Config
from src.preprocessors.state_preprocessing import StateDataLoader
from src.api.models import (
    PredictionRequest,
    LakmoosPredictionRequest,
)  # This is used just for easier implementation, can use Dict or implement custom pydantic class instead


# Get settings
settings = Config()

BASE_URL: str = f"http://{settings.api_host}:{settings.api_port}"
AVAILABLE_ENDPOINTS: Dict[str, str] = {
    "info": "/info",
    "predict": "/predict",
    "lakmoos-predict": "/lakmoos-predict",
}


def send_info_request() -> requests.Response:
    response = requests.get(url=f"{BASE_URL}{AVAILABLE_ENDPOINTS['info']}")
    return response


def send_base_prediction_request(
    state: str,
    model_key: str,
    target_year: int,
) -> requests.Response:

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
        url=f"{BASE_URL}{AVAILABLE_ENDPOINTS['predict']}",
        headers=headers,
        json=request.model_dump(),
    )

    return response


def send_lakmoos_prediction_request(
    state: str,
    model_key: str,
    target_year: int,
    max_age: int = 100,
) -> requests.Response:
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

    request = LakmoosPredictionRequest(
        model_key=model_key,
        state=state,
        input_data=input_data,
        target_year=target_year,
        max_age=max_age,
    )

    response = requests.post(
        url=f"{BASE_URL}{AVAILABLE_ENDPOINTS['lakmoos-predict']}",
        headers=headers,
        json=request.model_dump(),
    )

    return response
