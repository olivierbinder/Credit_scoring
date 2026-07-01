# src/credit_scoring/serving/inference.py
# %%  IMPORTS                                                                          .

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml

from credit_scoring.config import (
    EDUCATION_MAP,
    GENDER_MAP,
    PROD_MODEL,
    PROD_REFERENCE,
)
from credit_scoring.logger import logger
from credit_scoring.serving.schemas import CreditScoringInput


# %%  ARTIFACTS                                                                        .
_model = None
_features = None
_reference_df = None
_threshold = None


def get_model():
    global _model, _features, _threshold
    if _model is None:
        _model = mlflow.lightgbm.load_model(PROD_MODEL)
        _features = _model.feature_name_  # booster_.feature_name()
        with open(PROD_MODEL / "MLmodel") as f:
            mlmodel = yaml.safe_load(f)
        _threshold = mlmodel["metadata"]["optimal_threshold"]
        logger.info("🆗 Model loaded")
    return _model, _features, _threshold


def get_reference_df():
    global _reference_df
    if _reference_df is None:
        _reference_df = pd.read_parquet(PROD_REFERENCE)
    return _reference_df


_onnx_session = None


def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        onnx_path = str(PROD_MODEL / "model.onnx")
        _onnx_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
    return _onnx_session


# %%  FEATURE TRANSFORMATIONS                                                          .
def encode(input_data: CreditScoringInput) -> dict:
    features = input_data.model_dump()

    features["CODE_GENDER"] = GENDER_MAP[features["CODE_GENDER"]]
    features["NAME_EDUCATION_TYPE"] = EDUCATION_MAP[features["NAME_EDUCATION_TYPE"]]

    return features


# %%  DATA ACCESS                                                          .
def lookup(sk_id: int) -> dict | None:
    reference = get_reference_df()
    row = reference.loc[reference["SK_ID_CURR"] == sk_id]

    if row.empty:
        return None

    return {
        key: (None if pd.isna(value) else value)
        for key, value in row.iloc[0].to_dict().items()
    }


# %%  INFERENCE                                                          .
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


def predict_onnx(input_data: CreditScoringInput) -> dict:
    _, expected_features, threshold = get_model()
    session = get_onnx_session()

    features = encode(input_data)
    X = pd.DataFrame([features])[expected_features]
    X = X.replace({None: np.nan}).astype(np.float32)  # ONNX exige float32

    input_name = session.get_inputs()[0].name
    proba = session.run(None, {input_name: X.values})[1]  # [1] = probas, [0] = classes
    probability = float(proba[0][1])

    return {
        "probability": round(probability, 4),
        "prediction": (
            "Likely to default" if probability >= threshold else "Not likely to default"
        ),
    }
