from typing import Any, List, Optional
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class DataInputSchema(BaseModel):
    step: int
    type: str
    amount: float
    oldBalanceOrig:float
    newBalanceOrig:float
    oldBalanceDest: float
    newBalanceDest: float


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
