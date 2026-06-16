#!/bin/bash
set -e

uv run python -m credit_scoring.pipelines.training \
    --run-cv \
    --run-feat-imp \
    --run-shap \
    --max-rows 100000