from typing import Dict, List
from pydantic import BaseModel


# TODO:
# Implement interface for data transfer
class Info(BaseModel):
    avialable_models: List[str]


class Predict(BaseModel):
    state: str
    input_data: Dict
    target_year: int
