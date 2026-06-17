FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Installation des dépendances système nécessaires à LightGBM (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copie uniquement les fichiers de dépendances pour profiter du cache Docker
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Copie le reste (code + data)
COPY src/ ./src/
COPY data/processed/reference.parquet ./data/processed/reference.parquet
COPY mlruns/1/models/m-54d19f1191184fa4bae260eedb368fa1/artifacts/ ./models/production/

# Rendre le script exécutable
RUN chmod +x start.sh

EXPOSE 7860 8000

CMD ["./start.sh"]