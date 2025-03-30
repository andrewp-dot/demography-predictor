from typing import Dict, List, Optional
from pydantic import BaseModel


# TODO:
# Implement interface for data transfer
class Info(BaseModel):
    avialable_models: List[str]


class PredictionRequest(BaseModel):
    model_key: str
    state: str
    input_data: List[Dict]  # Each dict is one row of data
    target_year: int = 2030  # Set the default value for optional parameter

    # Maybe this can change into the standalone
    # 'lakmoos-prediction' or 'generator-prediction' or 'probability-prediction' flag
    # Reason: the output should not be human readable but id should be something like `gaussian curve` of probabilites for generators
    # return_probability: bool


class PredictionResponse(BaseModel):
    state: str
    predictions: List[Dict]


class LakmoosPredictionResponse(PredictionResponse):
    distribution: List[Dict]
