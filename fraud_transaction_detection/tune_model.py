from sklearn.model_selection import cross_val_score, train_test_split
import yaml
import optuna
from sklearn.ensemble import RandomForestClassifier
from fraud_transaction_detection.pipeline import fraud_detection_pipeline
from fraud_transaction_detection import config
from fraud_transaction_detection.config.core import set_config_to_yaml
from fraud_transaction_detection.processing.data_manager import get_training_data, load_dataset

def objective(trial):
    X_train, X_test, y_train, y_test = get_training_data()
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    
    pipeline = fraud_detection_pipeline.set_params(
        regressor__n_estimators=n_estimators,
        regressor__max_depth=max_depth,
        regressor__min_samples_split=min_samples_split,
        regressor__min_samples_leaf=min_samples_leaf,
    )

    score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1').mean()
    print("Score: ", score)
    return score  

# Optuna optimization
def tune():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    best_params = study.best_params

    # Read the current config

    # Update the config with the best hyperparameters
    config.model_config_.best_hyperparameters = best_params

    # Write the updated config back to the config.yml file
    set_config_to_yaml(config)
    print(f"Best Hyperparameters: {best_params}")

# Run the tuning process
if __name__ == "__main__":
    tune()
