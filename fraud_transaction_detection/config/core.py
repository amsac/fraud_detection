import yaml
from pathlib import Path
from typing import Dict, List
import fraud_transaction_detection
from pydantic import BaseModel
from strictyaml import YAML, load


PACKAGE_ROOT = Path(fraud_transaction_detection.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    training_data_file: str
    pipeline_save_file: str

class Hyperparameters(BaseModel):
    """
    Hyperparameters for the model.
    """
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    target: str
    features: List[str]
    test_size: float
    random_state: int
    type_var: str
    type_mapping: Dict[str, int]
    unused_fields: List[str]
    best_hyperparameters: Hyperparameters

class Config(BaseModel):
    """Master config object."""

    app_config_: AppConfig
    model_config_: ModelConfig

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def set_config_to_yaml(config, cfg_path: Path = None) -> None:
    """Set the config to a YAML file."""
    if not cfg_path:
        cfg_path = find_config_file()
    config_data = config.model_dump()
    with open(cfg_path, "w") as conf_file:
        yaml.dump(config_data, conf_file,sort_keys=False)


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    app_cfg_data = parsed_config.data["app_config_"]
    model_cfg_data = parsed_config.data["model_config_"]
    _config = Config(
        app_config_=AppConfig(**app_cfg_data),
        model_config_=ModelConfig(**model_cfg_data),
    )

    return _config


config = create_and_validate_config()