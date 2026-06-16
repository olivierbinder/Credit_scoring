"""
INFERENCE PIPELINE

This module contains the inference pipeline for credit scoring.

This pipeline:
1. Load MLflow-logged model and feature metadata from training
2. Receive validated input data via Pydantic model
3. Calls the inference pipeline to transform features and predict
4. Returns prediction in JSON format

Expected Response:
- {"prediction": "Likely to default"} or {"prediction": "Not likely to default"}
- {"probability": 0.5}




# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import mlflow

from credit_scoring.config import DIR_MODEL

# MODEL LOAD
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
try:
    model = mlflow.pyfunc.load_model(DIR_MODEL)
    print(f"Model loaded successfully from {DIR_MODEL}")

except Exception as e:
    raise Exception(f"Error loading model from {DIR_MODEL}: {e}")

# FEATURES
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
try:
    expected_features = model.artifacts["feature_names.txt"]
    print(f"Features loaded successfully from {DIR_MODEL}")
except Exception as e:
    raise Exception(f"Error loading features from {DIR_MODEL}: {e}")

"""
