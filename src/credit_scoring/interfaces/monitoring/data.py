# src/credit_scoring/interfaces/monitoring/data.py
"""
Shared data-loading helpers for the monitoring pages.
All loaders are @st.cache_data so they execute once per session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "PAYMENT_RATE",
    "OWN_CAR_AGE",
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

CATEGORICAL_FEATURES = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

NULLABLE_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_3",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE",
    "ACTIVE_DAYS_CREDIT_MAX",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "CC_CNT_DRAWINGS_CURRENT_VAR",
]

# Default paths — overridable via st.session_state set by the main app
DEFAULT_PRED_PATH = "logs/predictions.jsonl"
DEFAULT_API_PATH = "logs/api_calls.jsonl"
DEFAULT_REF_PATH = "data/processed/reference.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des logs de prédiction…")
def load_predictions(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    records = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None
    df = pd.DataFrame(records)
    inputs = pd.json_normalize(df["inputs"])
    df = pd.concat([df.drop(columns=["inputs"]), inputs], axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(show_spinner="Chargement des logs API…")
def load_api_calls(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    records = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(show_spinner="Chargement des données de référence…")
def load_reference(path: str, sample: int = 5000) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    df = df[cols]
    return df.sample(min(sample, len(df)), random_state=42)
