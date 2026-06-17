# src/credit_scoring/serving/inference.py
# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from credit_scoring.config import DIR_DATA_PROCESSED, PROD_MODEL_PATH
from credit_scoring.logger import logger
from credit_scoring.serving.constants import (
    EDUCATION_MAP,
    GENDER_MAP,
)

# MODEL ARTIFACTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
_model = None
_features = None
_reference_df = None
_threshold = None


def get_model():
    global _model, _features, _threshold
    if _model is None:
        _model = mlflow.lightgbm.load_model(PROD_MODEL_PATH)
        _features = _model.feature_name_  # booster_.feature_name()
        with open(PROD_MODEL_PATH / "MLmodel") as f:
            mlmodel = yaml.safe_load(f)
        _threshold = mlmodel["metadata"]["optimal_threshold"]
        logger.info("🆗 Model loaded")
    return _model, _features, _threshold


def get_reference_df():
    global _reference_df
    if _reference_df is None:
        _reference_df = pd.read_parquet(DIR_DATA_PROCESSED / "reference.parquet")
    return _reference_df


# INPUT SCHEMA
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class CreditScoringInput(BaseModel):
    # External scores
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_2: float = Field(..., ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1)

    # Application financials
    AMT_ANNUITY: float = Field(..., gt=0)
    AMT_GOODS_PRICE: float = Field(..., gt=0)
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int | None = None

    # Engineered features
    PAYMENT_RATE: float = Field(..., gt=0)
    OWN_CAR_AGE: float | None = Field(None, ge=0)

    CODE_GENDER: Literal["M", "F"]

    NAME_EDUCATION_TYPE: Literal[
        "Lower secondary",
        "Secondary / secondary special",
        "Incomplete higher",
        "Higher education",
        "Academic degree",
    ]

    # Installments
    INSTAL_DPD_MEAN: float = Field(..., ge=0)
    INSTAL_AMT_PAYMENT_SUM: float = Field(..., ge=0)

    # POS cash
    POS_CNT_INSTALMENT_FUTURE_MEAN: float = Field(..., ge=0)
    POS_SK_DPD_DEF_MEAN: float = Field(..., ge=0)

    # Previous applications
    PREV_CNT_PAYMENT_MEAN: float = Field(..., ge=0)
    PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN: float

    # Bureau
    ACTIVE_DAYS_CREDIT_MAX: float | None = None

    # Credit card
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: float | None = Field(None, ge=0)
    CC_CNT_DRAWINGS_CURRENT_VAR: float | None = Field(None, ge=0)


# FEATURE TRANSFORMATIONS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def encode(input_data: CreditScoringInput) -> dict:
    features = input_data.model_dump()

    features["CODE_GENDER"] = GENDER_MAP[features["CODE_GENDER"]]
    features["NAME_EDUCATION_TYPE"] = EDUCATION_MAP[features["NAME_EDUCATION_TYPE"]]

    return features


# DATA ACCESS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def lookup(sk_id: int) -> dict | None:
    reference = get_reference_df()
    row = reference.loc[reference["SK_ID_CURR"] == sk_id]

    if row.empty:
        return None

    return {
        key: (None if pd.isna(value) else value)
        for key, value in row.iloc[0].to_dict().items()
    }


# INFERENCE
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def predict(
    input_data: CreditScoringInput,
) -> dict:
    model, expected_features, threshold = get_model()
    features = encode(input_data)

    X = pd.DataFrame([features])[expected_features]

    X = X.replace({None: np.nan})
    X = X.astype(float)

    probability = float(model.predict_proba(X)[0, 1])

    return {
        "probability": round(probability, 4),
        "prediction": (
            "Likely to default" if probability >= threshold else "Not likely to default"
        ),
    }
