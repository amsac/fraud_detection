from sklearn.pipeline import Pipeline
from fraud_transaction_detection.config.core import config
from fraud_transaction_detection.processing.features import Mapper
from sklearn.ensemble import RandomForestClassifier

fraud_detection_pipeline = Pipeline(
    [
        (
            "type_mapping",
            Mapper(config.model_config_.type_var, config.model_config_.type_mapping),
        ),
        (
            "regressor",
            RandomForestClassifier(
                n_estimators=config.model_config_.best_hyperparameters.n_estimators,
                max_depth = config.model_config_.best_hyperparameters.max_depth,
                min_samples_leaf = config.model_config_.best_hyperparameters.min_samples_leaf,
                min_samples_split= config.model_config_.best_hyperparameters.min_samples_split
            ),
        ),
    ]
)
