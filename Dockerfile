FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
    # PYTHONPATH="/app/src"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dépendances d'abord — couche cacheable
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Code source
COPY src/ ./src/

RUN uv sync --frozen --no-dev

# Configs et seuils
COPY config/ ./config/
COPY reports/ ./reports/
COPY logs/ ./logs/

EXPOSE 7860 8000

# Start FastAPI in the background and Streamlit in the foreground
CMD ["sh", "-c", "uvicorn credit_scoring.serving.api:app --host 0.0.0.0 --port 8000 & exec streamlit run src/credit_scoring/interfaces/app_streamlit.py --server.port 7860 --server.address 0.0.0.0"]