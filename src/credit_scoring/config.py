# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# SETTINGS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 50)
RANDOM_STATE: int = 42
DTYPE_COLORS: dict[str, str] = {
    "float64": "#72efdd",
    "int64": "#64c6df",
    "object": "#ffb5a7",
    "bool": "#f5fbb9",
}


# PATHS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Directories
DIR_ROOT = Path(__file__).resolve().parents[2]
DIR_DATA = DIR_ROOT / "data"
DIR_DATA_RAW = DIR_DATA / "raw"
DIR_DATA_PROCESSED = DIR_DATA / "processed"
DIR_CONFIG = DIR_ROOT / "config"

# Desciption file
FILE_DESC = DIR_DATA / "description" / "HomeCredit_columns_description.csv"

# Raw files
FILE_APP_TEST = DIR_DATA_RAW / "application_test.csv"
FILE_APP_TRAIN = DIR_DATA_RAW / "application_train.csv"
FILE_BUREAU = DIR_DATA_RAW / "bureau.csv"
FILE_BUREAU_BALANCE = DIR_DATA_RAW / "bureau_balance.csv"
FILE_PREVIOUS_APP = DIR_DATA_RAW / "previous_application.csv"
FILE_PREVIOUS_BALANCE_CARD = DIR_DATA_RAW / "credit_card_balance.csv"
FILE_PREVIOUS_BALANCE_CASH = DIR_DATA_RAW / "POS_CASH_balance.csv"
FILE_PREVIOUS_PAYMENTS = DIR_DATA_RAW / "installments_payments.csv"

# Processed files
X_TRAIN_PROC_PATH = DIR_DATA_PROCESSED / "X_train_proc.csv"
X_TEST_PROC_PATH = DIR_DATA_PROCESSED / "X_test_proc.csv"
Y_TRAIN_PROC_PATH = DIR_DATA_PROCESSED / "y_train.csv"
Y_TEST_PROC_PATH = DIR_DATA_PROCESSED / "y_test.csv"
DF_PROC_PATH = DIR_DATA_PROCESSED / "df_proc.parquet"
PIPELINE_PATH = DIR_DATA_PROCESSED / "full_pipeline.pkl"


# MODELS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

PROD_MODEL = DIR_ROOT / "artifacts/production_model/model.pkl"

MODEL_REGISTRY = {
    "dummy": {
        "model_class": DummyClassifier,
        "default_params": {"strategy": "prior", "random_state": 42},
    },
    "logreg": {
        "model_class": LogisticRegression,
        "default_params": {
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 500,
            "random_state": 42,
        },
    },
    "random_forest": {
        "model_class": RandomForestClassifier,
        "default_params": {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
        },
    },
    "lightgbm": {
        "model_class": LGBMClassifier,
        "default_params": {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
            "importance_type": "gain",
        },
    },
    "xgboost": {
        "model_class": XGBClassifier,
        "default_params": {
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
            "eval_metric": "logloss",
        },
    },
    "catboost": {
        "model_class": CatBoostClassifier,
        "default_params": {
            "auto_class_weights": "Balanced",
            "random_state": 42,
            "verbose": 0,  # 0 pour ne rien afficher, 100 pour loguer tous les 100 arbres
            "allow_writing_files": False,
        },
    },
}

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# === Feature Names ===
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# From raw files


# Engineered Features

# Target Features

# Features Groups

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# === Source Mappings ===
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
# MAP_TRANSACTIONS = {}


"""
import sys
import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from credit_scoring.logger import logger

PROJ_ROOT = Path(__file__).resolve().parents[2]

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJ_ROOT / ".env", extra="ignore")
    RANDOM_STATE: int = 42
    PATHS: Dict[str, Any] = Field(default_factory=dict)
    COLUMNS: Dict[str, Any] = Field(default_factory=dict)
    PARAMS: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        try:
            instance = cls()
            config_dir = PROJ_ROOT / "config"
            def _read_yaml(name: str) -> dict:
                path = config_dir / f"{name}.yaml"
                return yaml.safe_load(path.read_text()) if path.exists() else {}

            p_yml = _read_yaml("project")
            f_yml = _read_yaml("features")
            m_yml = _read_yaml("model")

            raw = PROJ_ROOT / p_yml.get("directories", {}).get("raw", "data/raw")
            instance.PATHS = {
                "RAW": raw,
                "PROCESSED": PROJ_ROOT / p_yml.get("directories", {}).get("processed", "data/processed")
            }
            for k, v in p_yml.get("files", {}).items():
                instance.PATHS[k.upper()] = raw / v

            instance.COLUMNS = f_yml.get("columns", {})
            instance.PARAMS = m_yml.get("params", {})
            instance.RANDOM_STATE = p_yml.get("random_state", 42)
            logger.info("✅ Configuration loaded.")
            return instance
        except Exception as e:
            print(f"❌ Config Error: {e}")
            sys.exit(1)

settings = Config.load()
"""
