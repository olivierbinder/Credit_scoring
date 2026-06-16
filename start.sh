#!/bin/bash

# Lance l'API FastAPI en arrière-plan
uv run --no-dev uvicorn credit_scoring.interfaces.api:app_simple --host 0.0.0.0 --port 8000 &

# Lance Streamlit en premier plan (pour garder le conteneur vivant)
uv run --no-dev streamlit run src/credit_scoring/interfaces/app_streamlit.py --server.port 7860


uv run streamlit run src/credit_scoring/interfaces/app_streamlit.py