---
title: Credit Scoring
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Credit Scoring - ML Application

## Purpose

This project predicts credit default risk from Home Credit data and exposes the
model through a small production-style stack: a FastAPI backend, a Streamlit
application, monitoring views, and CI/CD deployment to Hugging Face Spaces.

The current app is focused on serving and monitoring a selected production model.
Some training options are configurable in code and YAML, but they are not exposed
as a fully dynamic product interface.

## Data And Model Scope

The preprocessing pipeline combines the main application data with historical
credit and previous application tables:

- application train/test data;
- bureau and bureau balance history;
- previous applications, installments, POS cash and credit card balance data.

The production model currently served by the API is a LightGBM classifier using
a reduced set of 20 selected features. The training code can instantiate several
model families from configuration, including LightGBM, XGBoost, CatBoost, random
forest, logistic regression and a dummy baseline.

## Key Features

- **Training pipeline:** YAML-driven experiment setup with preprocessing,
  train/test split, threshold optimisation and MLflow logging.
- **Feature selection:** stability-oriented feature ranking using repeated
  LightGBM importance over folds and random seeds.
- **FastAPI serving:** prediction, client lookup, model threshold and
  reference-data routes, with documented errors in Swagger.
- **Streamlit scoring UI:** load a client, edit the selected model features,
  run predictions and compare each value with the reference distribution.
- **Streamlit monitoring UI:** API route monitoring, latency/error summaries,
  Evidently data drift and data quality reports.
- **Inference optimisation:** ONNX export and Streamlit benchmark comparing
  standard LightGBM inference with ONNX Runtime.
- **CI/CD:** GitHub Actions run Ruff and Pytest, then deploy to Hugging Face
  Spaces after a successful CI run on `main`.

## Current Limitations

- The Streamlit app does not let users dynamically choose data sources or train
  new models from the UI.
- The deployed app serves the packaged production model and reference parquet files.
- The broader model registry exists in training code, but the current serving
  path is built around the selected LightGBM model.

## Project Structure

```text
src/credit_scoring/
|-- features/      # preprocessing and feature selection
|-- models/        # training, evaluation, tuning and explainability
|-- serving/       # FastAPI app, inference code and packaged model assets
`-- interfaces/    # Streamlit prediction and monitoring pages

scripts/
|-- export_onnx.py
|-- generate_base_for_inference.py
|-- run_ft_selection_nb.py
`-- run_ft_selection_ranking.py
```

## Main Commands

```bash
just api          # FastAPI on port 8000
just dashboard    # Streamlit app
just app          # API and dashboard together
just train        # training pipeline
just export-onnx  # export the production model to ONNX
just test         # test suite
```

## Tech Stack

- **Backend:** FastAPI, Pydantic, Uvicorn.
- **Frontend:** Streamlit and Plotly.
- **ML:** Pandas, Scikit-learn, LightGBM, ONNX Runtime, MLflow.
- **Monitoring:** API JSONL logs, Evidently reports, Streamlit dashboards.
- **Automation:** Ruff, Pytest, GitHub Actions, Docker and Hugging Face Spaces.
