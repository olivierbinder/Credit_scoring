# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from credit_scoring.config import DIR_DATA_PROCESSED, MLFLOW_TRACKING_URI

# MODEL ARTIFACTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.lightgbm.load_model("models:/Production/latest")
EXPECTED_FEATURES = model.feature_name_

model_info = mlflow.models.get_model_info("models:/Production/latest")
run = mlflow.get_run(model_info.run_id)

OPTIMAL_THRESHOLD = float(run.data.params["evaluation_threshold"])

REFERENCE_DF = pd.read_parquet(DIR_DATA_PROCESSED / "reference.parquet")

# REFERENCE DATA
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
CATEGORICAL_FEATURES = {
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
}

FEATURE_LABELS = {
    "EXT_SOURCE_1": "External Score 1",
    "EXT_SOURCE_2": "External Score 2",
    "EXT_SOURCE_3": "External Score 3",
    "CODE_GENDER": "Gender",
    "NAME_EDUCATION_TYPE": "Education",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "OWN_CAR_AGE": "Car Age",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_GOODS_PRICE": "Goods Price",
    "PAYMENT_RATE": "Payment Rate",
    "INSTAL_DPD_MEAN": "Avg Days Past Due",
    "INSTAL_AMT_PAYMENT_SUM": "Installment Payments",
    "POS_CNT_INSTALMENT_FUTURE_MEAN": "Future Installments",
    "POS_SK_DPD_DEF_MEAN": "POS Delinquency",
    "PREV_CNT_PAYMENT_MEAN": "Previous Payment Count",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN": "Previous Due Date",
    "ACTIVE_DAYS_CREDIT_MAX": "Active Credit Age",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": "ATM Withdrawals",
    "CC_CNT_DRAWINGS_CURRENT_VAR": "Card Usage Variability",
}


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


# CATEGORICAL ENCODING
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
GENDER_MAP = {
    "M": 1.0,
    "F": 0.0,
}

EDUCATION_MAP = {
    "Lower secondary": 0.0,
    "Secondary / secondary special": 1.0,
    "Incomplete higher": 2.0,
    "Higher education": 3.0,
    "Academic degree": 4.0,
}

GENDER_INVERSE = {v: k for k, v in GENDER_MAP.items()}
EDUCATION_INVERSE = {v: k for k, v in EDUCATION_MAP.items()}


# FEATURE TRANSFORMATIONS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def encode(input_data: CreditScoringInput) -> dict:
    features = input_data.model_dump()

    features["CODE_GENDER"] = GENDER_MAP[features["CODE_GENDER"]]
    features["NAME_EDUCATION_TYPE"] = EDUCATION_MAP[features["NAME_EDUCATION_TYPE"]]

    return features


def decode_for_display(raw: dict) -> dict:
    decoded = raw.copy()

    decoded["CODE_GENDER"] = GENDER_INVERSE.get(
        raw["CODE_GENDER"],
        "M",
    )

    decoded["NAME_EDUCATION_TYPE"] = EDUCATION_INVERSE.get(
        raw["NAME_EDUCATION_TYPE"],
        "Higher education",
    )

    return decoded


# DATA ACCESS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def lookup(sk_id: int) -> dict | None:
    row = REFERENCE_DF.loc[REFERENCE_DF["SK_ID_CURR"] == sk_id]

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
    threshold: float = OPTIMAL_THRESHOLD,
) -> dict:
    features = encode(input_data)

    X = pd.DataFrame([features])[EXPECTED_FEATURES]

    X = X.replace({None: np.nan})
    X = X.astype(float)

    probability = float(model.predict_proba(X)[0, 1])

    return {
        "probability": round(probability, 4),
        "prediction": (
            "Likely to default" if probability >= threshold else "Not likely to default"
        ),
    }
