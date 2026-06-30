# src/credit_scoring/serving/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


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


class ErrorResponse(BaseModel):
    detail: str | list[dict]
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    threshold: float


class PredictionResponse(BaseModel):
    probability: float
    prediction: str
