# src/credit_scoring/serving/api.py

# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import traceback

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from credit_scoring.serving.inference import (
    CreditScoringInput,
    get_model,
    get_reference_df,
    lookup,
    predict,
)

# APP INIT
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
app = FastAPI(
    title="Credit Scoring API ",
    description="ML API to predict credit risk",
    version="0.1.0",
)


# HEALTH CHECK ENDPOINT
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@app.get("/")
def root():
    """Health check endpoint for monitoring and load balancer health checks."""
    return {"status": "ok"}


# MAIN PREDICTION API ENDPOINT
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@app.get("/model-info")
def model_info():
    _, _, threshold = get_model()
    return {"threshold": threshold}


def _serialize_row(row: dict) -> dict:
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


@app.get("/reference")
def get_reference_data():
    df = get_reference_df()
    return Response(
        content=df.to_parquet(index=False),
        media_type="application/octet-stream",
    )


@app.get("/lookup/{sk_id}")
def lookup_client(sk_id: int):
    features = lookup(sk_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id} not found")
    return features


@app.post("/predict")
def predict_client(input_data: CreditScoringInput):
    try:
        return predict(input_data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
