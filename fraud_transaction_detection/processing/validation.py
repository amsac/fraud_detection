import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from pydantic import BaseModel, ValidationError
from fraud_transaction_detection.config.core import config
from fraud_transaction_detection.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None
    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()
    return validated_data, errors


class DataInputSchema(BaseModel):
    step: int
    type: str
    amount: float
    oldBalanceOrig:float
    newBalanceOrig:float
    oldBalanceDest: float
    newBalanceDest: float

input_data = [{
    "step": "winter",
    "type":"2012-11-05",
    "amount": "6am",
    "oldBalanceOrig": "No",
    "newBalanceOrig": "Sun",
    "oldBalanceDest": "Yes",
    "newBalanceDest": "Mist",
}]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]


