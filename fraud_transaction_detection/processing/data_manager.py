from typing import List
import joblib
import os
import pandas as pd
from pathlib import Path
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fraud_transaction_detection import __version__ as _version  # noqa: F401
from fraud_transaction_detection.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # If __file__ doesn't exist (e.g., in a notebook), fallback to cwd
    BASE_DIR = Path.cwd()

# BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
    if hasattr(config, "unused_fields"):
        data_frame = data_frame.drop(columns=config.model_config_.unused_fields, errors='ignore')
    return data_frame

def download_and_save_dataset(save_as: str = "fraud_data.csv") -> None:
    save_path = DATASET_DIR / save_as

    # Only download if file doesn't already exist
    if save_path.exists():
        print(f"Dataset already exists at: {save_path}")
        return

    print("Downloading dataset...")
    path = kagglehub.dataset_download("ealaxi/paysim1")
    df = pd.read_csv(Path(path) / "PS_20174392719_1491204439457_log.csv")
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to: {save_path}")

def download_data(*, file_name: str = "fraud_data.csv") -> pd.DataFrame:
    file_path = DATASET_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    return pd.read_csv(file_path)


def load_dataset(*, file_name: str ) -> pd.DataFrame:
    download_and_save_dataset("fraud_data.csv")
    df = download_data(file_name="fraud_data.csv")
    # print(f"Dataset saved to: {DATASET_DIR / save_as}")
    df = save_for_testing_unseen_data(df)
    fraud_df = df[df['isFraud'] == 1]
    non_fraud_df = df[df['isFraud'] == 0]
    non_fraud_sample = non_fraud_df.sample(n=len(fraud_df)*5, random_state=42) 
    balanced_df = pd.concat([fraud_df, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print("training data shape...", balanced_df.shape)
    transformed = pre_pipeline_preparation(data_frame=balanced_df)
    return transformed

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def get_training_data() -> pd.DataFrame:
    """
    Fetch the training data from the dataset directory.
    """
    data = load_dataset(file_name=config.app_config_.training_data_file)
    features = config.model_config_.features

    X_train, X_test, y_train, y_test = train_test_split(
        data[features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
    )
    print("X_train shape",X_train.shape)
    print("X_test shape",X_test.shape)      
    print("y_train shape",y_train.shape)
    print("y_test shape",y_test.shape)
    return X_train, X_test, y_train, y_test

def save_for_testing_unseen_data(df: pd.DataFrame) -> pd.DataFrame:
    test_rows_0 = df[df["isFraud"] == 0].sample(n=2, random_state=42)
    test_rows_1 = df[df["isFraud"] == 1].sample(n=2, random_state=42)
    test_rows = pd.concat([test_rows_0, test_rows_1])
    test_rows.to_csv(DATASET_DIR / "test_rows.csv", index=False)
    df = df.drop(index=test_rows.index)
    return df
    
