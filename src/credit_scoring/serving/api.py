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
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from credit_scoring.config import FILE_API, FILE_PRED
from credit_scoring.serving.inference import (
    get_model,
    get_reference_df,
    lookup,
    predict,
)
from credit_scoring.serving.schemas import (
    CreditScoringInput,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

PROCESS = psutil.Process(os.getpid())


COMMON_ERROR_RESPONSES = {
    500: {
        "model": ErrorResponse,
        "description": "Unexpected server error. Check the request_id in API logs.",
    },
}


# %%  LOGGING SETUP                                                                    .
def append_log(path: Path, record: dict) -> None:
    """Append a JSON record to a .jsonl log file (one JSON object per line)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# %%  APP INIT                                                                         .
app = FastAPI(
    title="Credit Scoring API",
    description=(
        "ML API to predict credit risk.\n\n"
        "Errors use a common JSON format with `detail` and `request_id` so they "
        "can be matched with entries in the API logs."
    ),
    version="0.1.0",
)


# %%  ERROR HANDLERS                                                                   .
def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": jsonable_encoder(exc.detail),
            "request_id": _request_id(request),
        },
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": jsonable_encoder(exc.errors()),
            "request_id": _request_id(request),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": _request_id(request),
        },
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
    append_log(FILE_API, record)

    # Forward the request ID for client-side correlation
    response.headers["X-Request-Id"] = request_id
    return response


# %%  HEALTH CHECK ENDPOINT                                                            .
@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    responses=COMMON_ERROR_RESPONSES,
)
def root():
    """Health check endpoint for monitoring and load balancer health checks."""
    return {"status": "ok"}


# %% MODEL INFO                                                                        .
@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get model metadata",
    responses=COMMON_ERROR_RESPONSES,
)
def model_info():
    _, _, threshold = get_model()
    return {"threshold": threshold}


# %% REFERENCE DATA                                                                    .
def _serialize_row(row: dict) -> dict:
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


@app.get(
    "/reference",
    summary="Download reference data",
    responses={
        200: {
            "description": "Reference dataset as a parquet binary payload.",
            "content": {"application/octet-stream": {}},
        },
        **COMMON_ERROR_RESPONSES,
    },
)
def get_reference_data():
    df = get_reference_df()
    return Response(
        content=df.to_parquet(index=False),
        media_type="application/octet-stream",
    )


@app.get(
    "/lookup/{sk_id}",
    summary="Look up a client by ID",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Client ID not found in the reference dataset.",
        },
        **COMMON_ERROR_RESPONSES,
    },
)
def lookup_client(sk_id: int):
    features = lookup(sk_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id} not found")
    return features


# %% PREDICTION ENDPOINT — with detailed prediction logging                            .
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict credit default risk",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid input payload. See detail for validation errors.",
        },
        **COMMON_ERROR_RESPONSES,
    },
)
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
        # Metrics before inference
        mem_before_mb = PROCESS.memory_info().rss / 1024**2

        # Measure model inference only
        inference_start = time.perf_counter()

        result = predict(input_data)

        inference_ms = round(
            (time.perf_counter() - inference_start) * 1000,
            2,
        )

        # Metrics after inference
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

        append_log(FILE_PRED, log_record)

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

        append_log(FILE_PRED, log_record)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )
