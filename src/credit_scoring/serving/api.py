# src/credit_scoring/serving/api.py

# %%  IMPORTS                                                                          .
import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from credit_scoring.serving.inference import (
    CreditScoringInput,
    get_model,
    get_reference_df,
    lookup,
    predict,
)

PROCESS = psutil.Process(os.getpid())


# %%  LOGGING SETUP                                                                    .
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"
API_LOG = LOG_DIR / "api_calls.jsonl"


def append_log(path: Path, record: dict) -> None:
    """Append a JSON record to a .jsonl log file (one JSON object per line)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# %%  APP INIT                                                                         .
app = FastAPI(
    title="Credit Scoring API",
    description="ML API to predict credit risk",
    version="0.1.0",
)


# %%  MIDDLEWARE — API-LEVEL LOGGING (all routes)                                      .
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware that logs every HTTP request/response with timing.
    Covers: method, path, status code, latency_ms, request_id.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    is_error = response.status_code >= 400

    record = {
        "event": "api_call",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "is_error": is_error,
        "client_host": request.client.host if request.client else None,
    }
    append_log(API_LOG, record)

    # Propagate the request_id in response headers for client-side correlation
    response.headers["X-Request-Id"] = request_id
    return response


# %%  HEALTH CHECK ENDPOINT                                                            .
@app.get("/")
def root():
    """Health check endpoint for monitoring and load balancer health checks."""
    return {"status": "ok"}


# %% MODEL INFO                                                                        .
@app.get("/model-info")
def model_info():
    _, _, threshold = get_model()
    return {"threshold": threshold}


# %% REFERENCE DATA                                                                    .
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


# %% PREDICTION ENDPOINT — with detailed prediction logging                            .
@app.post("/predict")
async def predict_client(input_data: CreditScoringInput, request: Request):
    """
    Run credit scoring inference and log:
      - raw input features
      - model outputs
      - latency and inference time
      - CPU / memory usage
      - errors
    """
    request_id = request.state.request_id
    timestamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    log_record: dict = {
        "event": "prediction",
        "request_id": request_id,
        "timestamp": timestamp,
        "inputs": input_data.model_dump(),
        "probability": None,
        "prediction_label": None,
        "latency_ms": None,
        "inference_ms": None,
        "cpu_percent": None,
        "memory_mb": None,
        "memory_delta_mb": None,
        "model_version": "sklearn_v1",
        "success": False,
        "error": None,
    }

    try:
        # System metrics before inference
        mem_before_mb = PROCESS.memory_info().rss / 1024**2

        # Measure model inference only
        inference_start = time.perf_counter()

        result = predict(input_data)

        inference_ms = round(
            (time.perf_counter() - inference_start) * 1000,
            2,
        )

        # System metrics after inference
        cpu_percent = psutil.cpu_percent(interval=None)
        mem_after_mb = PROCESS.memory_info().rss / 1024**2

        log_record.update(
            {
                "probability": result["probability"],
                "prediction_label": result["prediction"],
                "latency_ms": round(
                    (time.perf_counter() - start) * 1000,
                    2,
                ),
                "inference_ms": inference_ms,
                "cpu_percent": cpu_percent,
                "memory_mb": round(mem_after_mb, 1),
                "memory_delta_mb": round(
                    mem_after_mb - mem_before_mb,
                    1,
                ),
                "success": True,
            }
        )

        append_log(PREDICTION_LOG, log_record)

        return result

    except Exception as e:
        traceback.print_exc()

        log_record.update(
            {
                "latency_ms": round(
                    (time.perf_counter() - start) * 1000,
                    2,
                ),
                "success": False,
                "error": str(e),
            }
        )

        append_log(PREDICTION_LOG, log_record)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
