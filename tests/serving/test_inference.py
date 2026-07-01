import numpy as np
import pandas as pd

from credit_scoring.serving import inference
from credit_scoring.serving.inference import encode, lookup, predict
from credit_scoring.serving.schemas import CreditScoringInput


def _make_input() -> CreditScoringInput:
    return CreditScoringInput(
        EXT_SOURCE_1=0.1,
        EXT_SOURCE_2=0.2,
        EXT_SOURCE_3=0.3,
        AMT_ANNUITY=1000.0,
        AMT_GOODS_PRICE=10000.0,
        DAYS_BIRTH=-10000,
        DAYS_EMPLOYED=-100,
        PAYMENT_RATE=0.1,
        OWN_CAR_AGE=2.0,
        CODE_GENDER="M",
        NAME_EDUCATION_TYPE="Higher education",
        INSTAL_DPD_MEAN=0.0,
        INSTAL_AMT_PAYMENT_SUM=100.0,
        POS_CNT_INSTALMENT_FUTURE_MEAN=1.0,
        POS_SK_DPD_DEF_MEAN=0.0,
        PREV_CNT_PAYMENT_MEAN=2.0,
        PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN=-10.0,
        ACTIVE_DAYS_CREDIT_MAX=-20.0,
        CC_CNT_DRAWINGS_ATM_CURRENT_MEAN=0.0,
        CC_CNT_DRAWINGS_CURRENT_VAR=0.0,
    )


def test_encode_maps_categorical_features():
    encoded = encode(_make_input())

    assert encoded["CODE_GENDER"] == 1.0
    assert encoded["NAME_EDUCATION_TYPE"] == 3.0
    assert encoded["AMT_ANNUITY"] == 1000.0


def test_lookup_returns_none_when_client_missing(monkeypatch):
    monkeypatch.setattr(
        inference,
        "get_reference_df",
        lambda: pd.DataFrame({"SK_ID_CURR": [1, 2], "VALUE": [10, 20]}),
    )

    assert lookup(999) is None


def test_lookup_converts_nan_to_none(monkeypatch):
    monkeypatch.setattr(
        inference,
        "get_reference_df",
        lambda: pd.DataFrame({"SK_ID_CURR": [1], "VALUE": [float("nan")]}),
    )

    result = lookup(1)
    assert result == {"SK_ID_CURR": 1, "VALUE": None}


def test_predict_uses_model_threshold(monkeypatch):
    class DummyModel:
        feature_name_ = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "PAYMENT_RATE",
            "OWN_CAR_AGE",
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "INSTAL_DPD_MEAN",
            "INSTAL_AMT_PAYMENT_SUM",
            "POS_CNT_INSTALMENT_FUTURE_MEAN",
            "POS_SK_DPD_DEF_MEAN",
            "PREV_CNT_PAYMENT_MEAN",
            "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
            "ACTIVE_DAYS_CREDIT_MAX",
            "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
            "CC_CNT_DRAWINGS_CURRENT_VAR",
        ]

        def predict_proba(self, X):
            return np.array([[0.25, 0.75]])

    monkeypatch.setattr(
        inference,
        "get_model",
        lambda: (DummyModel(), DummyModel.feature_name_, 0.5),
    )

    result = predict(_make_input())
    assert result == {"probability": 0.75, "prediction": "Likely to default"}
