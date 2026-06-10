import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from credit_scoring.config import PROD_MODEL

model = joblib.load(PROD_MODEL)
expected_features = model.feature_name_


class SimplePropertyFeatures(BaseModel):
    """Caractéristiques simples d'une propriété pour la prédiction"""

    EXT_SOURCE_3: float = 1.0
    EXT_SOURCE_2: float = 1.0
    EXT_SOURCE_1: float = 1.0
    PAYMENT_RATE: float = 1.0
    DAYS_EMPLOYED: float = 1000
    AMT_ANNUITY: float = 1.0
    AMT_GOODS_PRICE: float = 1.0
    DAYS_BIRTH: float = 1.0
    CODE_GENDER: float = 1.0
    INSTAL_DPD_MEAN: float = 1.0
    OWN_CAR_AGE: float = 3
    AMT_CREDIT: float = 1.0
    DAYS_ID_PUBLISH: float = 1.0
    INSTAL_AMT_PAYMENT_SUM: float = 1.0
    POS_CNT_INSTALMENT_FUTURE_MEAN: float = 1.0
    NAME_EDUCATION_TYPE: float = 1.0
    ANNUITY_INCOME_PERC: float = 10000
    PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN: float = 1.0
    REGION_POPULATION_RELATIVE: float = 1.0
    DAYS_EMPLOYED_PERC: float = 0.2


# Initialisation
app_simple = FastAPI(
    title="Credit Scoring API ",
    description="API to predict credit risk",
    version="0.1.0",
)


# Root endpoint
@app_simple.get("/")
def root():
    """Basic API status"""
    return {
        "message": "Welcome to the Credit Scoring",
        "status": "running",
    }


@app_simple.post("/predict")
async def predict(features: SimplePropertyFeatures):
    try:
        # Convert Pydantic object to dict
        data_dict = features.model_dump()

        # Create a DataFrame explicitly using the expected_features list as columns.
        # This forces pandas to align the data correctly with the model's structure.
        input_df = pd.DataFrame([data_dict], columns=expected_features)

        # Make the prediction
        prediction = model.predict(input_df)

        # Get probability for the positive class
        # Note: Index [1] is usually the 'risk/default' class
        probability = model.predict_proba(input_df)[0][1]
        result = {
            "prediction": int(prediction[0]),
            "probability": float(probability),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
