FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./

# Installation des dépendances
RUN uv sync --frozen --no-dev

# Copie le reste du projet et le script de démarrage
COPY . .
RUN chmod +x start.sh

EXPOSE 8000 7860

# Lance le script de démarrage
CMD ["./start.sh"]