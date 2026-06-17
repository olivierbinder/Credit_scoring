#!/bin/bash

uv run --no-project uvicorn credit_scoring.serving.api:app --host 0.0.0.0 --port 8000 --reload &
uv run --no-project streamlit run src/credit_scoring/interfaces/app_streamlit.py --server.port 7860 --server.runOnSave true
