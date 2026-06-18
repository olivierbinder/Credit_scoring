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

# Script de démarrage
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 7860 8000

CMD ["./start.sh"]
