#!/bin/bash

# Lance l'API FastAPI en arrière-plan
uv run uvicorn credit_scoring.interfaces.api:app_simple --host 0.0.0.0 --port 8000 &

# Lance Streamlit en premier plan (pour garder le conteneur vivant)
uv run streamlit run credit_scoring.interfaces.app_streamlit.py --server.port 7860