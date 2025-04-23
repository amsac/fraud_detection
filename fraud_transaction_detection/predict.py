from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from fraud_transaction_detection.pipeline import fraud_detection_pipeline
from fraud_transaction_detection.config.core import DATASET_DIR, config
from fraud_transaction_detection import __version__ as _version
from fraud_transaction_detection.processing.data_manager import load_pipeline, pre_pipeline_preparation
from fraud_transaction_detection.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
fraud_detection = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))   
    # validated_data = validated_data[config.model_config_.features]
    validated_data = validated_data.reindex(columns = config.model_config_.features)  
    results = {"predictions": None, "version": _version, "errors": errors}    
    if not errors:
        predictions = fraud_detection.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
    return results

if __name__ == "__main__":
    res = {0:"Normal", 1:"Fraud"}
    test_df = pd.read_csv(Path(f"{DATASET_DIR}/test_rows.csv"))
    test_df = pre_pipeline_preparation(data_frame=test_df)
    actual_values = test_df[config.model_config_.target].values
    feature_cols = config.model_config_.features
    input_data = test_df[feature_cols]
    predictions = make_prediction(input_data=input_data)
    for i, (pred, actual) in enumerate(zip(predictions["predictions"], actual_values)):
        print(f"Sample {i+1} -> Prediction: {res[pred]}, Actual: {res[actual]}")
