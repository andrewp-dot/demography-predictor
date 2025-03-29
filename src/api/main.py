# Standard library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict
import logging
import pandas as pd

from contextlib import asynccontextmanager

# Custom imports
from config import Config
from src.api.models import PredictionRequest, PredictionResponse
from src.utils.log import setup_logging
from src.utils.save_model import get_model
from src.predictors.predictor_base import DemographyPredictor

from src.api.models import Info


# Get the settings and logger
settings = Config()
logger = logging.getLogger("api")


LOADED_MODELS: Dict[str, DemographyPredictor] = {}


def load_models() -> None:
    # Load model(s) for prediction
    for model_key in settings.prediction_models.keys():
        try:
            LOADED_MODELS[model_key] = get_model(settings.prediction_models[model_key])
        except Exception as e:
            logger.error(f"Could not load the model: '{model_key}'. Reason: {str(e)}")


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
    input_df: pd.DataFrame = pd.DataFrame(request.input_data)

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
    prediction_df: pd.DataFrame = model.predict(
        input_data=input_df, last_year=LAST_YEAR, target_year=TARGET_YEAR
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
    return {"message": "Hello, FastAPI!"}


@app.get("/info")
def get_info():
    return Info(avialable_models=LOADED_MODELS.keys()).model_dump()


@app.post("/predict")
def model_predict(request: PredictionRequest) -> PredictionResponse:
    """
    Sends response to the PredictionRequest.

    Args:
        request (PredictionRequest): Request with the data for prediction generations.

    Returns:
        out: PredictionResponse: response for the prediction request.
    """
    logger.debug(f"Data: {request.input_data}")

    prediction_df = predict_from_request(request=request)

    # Process predictions
    # Convert prediction df to List[Dict]
    prediction_list = prediction_df.to_dict(orient="records")

    # Convert prediction dataframe to the list of dicts..
    return PredictionResponse(state=request.state, predictions=prediction_list)


@app.post("/lakmoos-predict")
def model_lakmoos_predict(request: PredictionRequest):

    # TODO: implemenet this using specification
    # From the prediction make probablity curve for the target year, the probability curve will serve as an inputs for generators
    # Decide of how many prediction curves will be the output, for example: aging_curve, gender_distribution_curve

    # Generate predictions
    prediction_df = predict_from_request(request=request)

    # Process predictions

    raise HTTPException(
        status_code=500, detail="Prediction for lakmoos not implemented yet!"
    )


def main(reload: bool = False):
    # Run API
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    main(reload=True)
