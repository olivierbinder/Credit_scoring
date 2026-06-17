#!/bin/bash
set -e

uv run python -m credit_scoring.models.training_pipeline \
    --run-cv \
    --run-feat-imp \
    --run-shap \