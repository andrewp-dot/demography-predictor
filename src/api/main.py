# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict, List
import logging
import pandas as pd

from contextlib import asynccontextmanager

# Custom imports
from config import Config
from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    LakmoosPredictionRequest,
    LakmoosPredictionResponse,
)
from src.api.distribution_curve import AgeDistribution

from src.utils.log import setup_logging
from src.utils.constants import aging_targets, gender_distribution_targets

from src.pipeline import PredictorPipeline
from src.preprocessors.state_preprocessing import StateDataLoader

from src.api.models import Info


# Get the settings and logger
settings = Config()
logger = logging.getLogger("api")


LOADED_MODELS: Dict[str, PredictorPipeline] = {}

AGING_COLUMNS: List[str] = gender_distribution_targets()

GENDER_DIST_COLUMNS: List[str] = aging_targets()


def load_models() -> None:
    # Load model(s) for prediction
    for model_key in settings.prediction_models.keys():
        try:
            LOADED_MODELS[model_key] = PredictorPipeline.get_pipeline(
                settings.prediction_models[model_key]
            )
        except Exception as e:
            logger.error(f"Could not load the model: '{model_key}'. Reason: {str(e)}")


def get_dataframe_from_request(
    state: str,
    # input_data: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Convert input data to pandas dataframe.

    Args:
        input_data (pd.DataFrame | None): Input data for prediction generation.

    Returns:
        out: pd.DataFrame: Converted input data.
    """
    # if input_data is None:
    # Load gathered data
    state_loader = StateDataLoader(state=state)
    input_df: pd.DataFrame = state_loader.load_data()
    # else:
    #     input_df: pd.DataFrame = pd.DataFrame(input_data)

    return input_df


def predict_from_request(request: PredictionRequest) -> pd.DataFrame:
    """
    Process request and generate prediction dataframe.

    Args:
        request (PredictionRequest): Request for prediction generation.

    Raises:
        HTTPException: If the 'year' is missing from columns raises HTTPException with status code `400`.
        HTTPException: If the last year from the given input sequence is greater or equal to the target year. Raises HTTPException with the status code of `400`.
        HTTPException: If the `model_key` is not recognized, raises HTTPException with the status code of `404`.

    Returns:
        out: pd.DataFrame: Generated prediction dataframe.
    """
    # 0. Create input dataframe and extract last and target year

    input_df = get_dataframe_from_request(
        state=request.state,
        # input_data=request.input_data,
    )

    # Verify if the 'year' columns is in the data
    if not "year" in input_df.columns:
        raise HTTPException(
            status_code=400, detail="The input data are not in appropriate format. "
        )

    # Sort data
    input_df = input_df.sort_values(by=["year"], ascending=True)

    LAST_YEAR = int(input_df[["year"]].iloc[-1].item())
    TARGET_YEAR = request.target_year

    # Check if the target year is greater then last year
    if LAST_YEAR >= TARGET_YEAR:
        raise HTTPException(
            status_code=400,
            detail="The last year in data is less or equal to target year. Nothing to predict.",
        )

    # 1. Model selection
    try:
        model = LOADED_MODELS[request.model_key]
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=404,
            detail=f"Model with key '{request.model_key}' was not found.",
        )

    # 3. Prediction generation
    # prediction_df: pd.DataFrame = model.predict(
    #     input_data=input_df, last_year=LAST_YEAR, target_year=TARGET_YEAR
    # )

    prediction_df: pd.DataFrame = model.predict(
        input_data=input_df, target_year=TARGET_YEAR
    )

    return prediction_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    setup_logging()
    load_models()
    logger.info(f"Models loaded: {list(LOADED_MODELS.keys())}")

    yield

    # Cleanup
    logger.info("Bye bye..")


# Define the app
app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello! If you want more info, just call '/info' endpoint."}


@app.get("/info")
def get_info():
    return Info(models=LOADED_MODELS.keys(), available_populations=[]).model_dump()


@app.post("/predict")
def model_predict(request: PredictionRequest) -> PredictionResponse:
    """
    Sends response to the PredictionRequest.

    Args:
        request (PredictionRequest): Request with the data for prediction generations.

    Returns:
        out: PredictionResponse: response for the prediction request.
    """
    # logger.debug(f"Data: {request.input_data}")

    prediction_df = predict_from_request(request=request)

    # Process predictions
    # Convert prediction df to List[Dict]
    prediction_list = prediction_df.to_dict(orient="records")

    # Convert prediction dataframe to the list of dicts..
    return PredictionResponse(state=request.state, predictions=prediction_list)


@app.post("/lakmoos-predict")
def model_lakmoos_predict(request: LakmoosPredictionRequest):

    # Generate predictions
    prediction_df = predict_from_request(request=request)
    print(prediction_df.columns)

    # If the output contains aging data
    if set(AGING_COLUMNS).issubset(set(prediction_df.columns)):

        # Create dict from the last record
        desired_years_prediction = prediction_df.iloc[-1].to_dict()
        pop_0_14 = desired_years_prediction[AGING_COLUMNS[0]]
        pop_15_64 = desired_years_prediction[AGING_COLUMNS[1]]
        pop_65_above = desired_years_prediction[AGING_COLUMNS[2]]

        curve_creator = AgeDistribution(
            pop_0_14=pop_0_14, pop_15_64=pop_15_64, pop_65_above=pop_65_above
        )

        # Get the curve
        age_probabilities_df = curve_creator.get_age_probabilities(
            max_age=request.max_age
        )

        # Send response
        return LakmoosPredictionResponse(
            state=request.state,
            predictions=prediction_df.to_dict(orient="records"),
            max_age=request.max_age,
            distribution=age_probabilities_df.to_dict(orient="records"),
        )
    elif set(GENDER_DIST_COLUMNS).issubset(set(prediction_df.columns)):

        # Adjust values to be coeficients / probabilities

        distribution_df = prediction_df.copy()

        # Convert percent to probability
        for col in GENDER_DIST_COLUMNS:
            distribution_df[col] = distribution_df[col] / 100

        # Get desired years
        desired_years_prediction = distribution_df.tail(1)

        return LakmoosPredictionResponse(
            state=request.state,
            predictions=prediction_df.to_dict(orient="records"),
            max_age=request.max_age,
            distribution=desired_years_prediction.to_dict(orient="records"),
        )

    raise HTTPException(
        status_code=500,
        detail=f"Prediction for Lakmoos of '{request.model_key}' not implemented yet!",
    )


def main(reload: bool = True):
    # Run API
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    main(reload=True)
