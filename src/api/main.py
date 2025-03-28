# Standard library imports
import uvicorn
from fastapi import FastAPI
from typing import Dict
import logging

from contextlib import asynccontextmanager

# Custom imports
from config import Config
from src.api.models import Predict
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


@app.post("/lakmoos-predict")
def read_root():
    return {"message": "Hello, FastAPI!"}


def main():
    # Run API
    uvicorn.run(
        "src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True
    )


if __name__ == "__main__":
    main()
