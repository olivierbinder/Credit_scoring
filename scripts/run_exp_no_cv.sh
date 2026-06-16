#!/bin/bash
set -e

uv run python -m credit_scoring.pipelines.training \
    --run-feat-imp \
    --run-shap \