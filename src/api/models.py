from typing import Dict, List, Optional
from pydantic import BaseModel


# TODO:
# Make info endpoint to gather data about endpoints and interfaces
class Info(BaseModel):
    avialable_models: List[str]
    # endpoints: List[str] = [
    #     "/predict",
    #     "/lakmoos-predict",
    #     "/info",
    # ]


class PredictionRequest(BaseModel):
    model_key: str
    state: str
    input_data: Optional[List[Dict]]  # Each dict is one row of data
    target_year: int = 2030  # Set the default value for optional parameter

    # Maybe this can change into the standalone
    # 'lakmoos-prediction' or 'generator-prediction' or 'probability-prediction' flag
    # Reason: the output should not be human readable but id should be something like `gaussian curve` of probabilites for generators
    # return_probability: bool


class LakmoosPredictionRequest(PredictionRequest):
    max_age: int = 100


class PredictionResponse(BaseModel):
    state: str
    predictions: List[Dict]


class LakmoosPredictionResponse(PredictionResponse):
    max_age: int
    distribution: List[Dict]
