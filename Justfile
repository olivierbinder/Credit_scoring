# Affiche les recettes disponibles
default:
    @just --list

# %% APP                                                                               .
# Lance l'API FastAPI
api:
    uv run uvicorn credit_scoring.serving.api:app --reload --port 8000

# Lance le dashboard Streamlit
dashboard:
    uv run streamlit run src/credit_scoring/interfaces/app_streamlit.py

# Lance les deux en parallèle
app:
    just api & just dashboard

# %% DOC                                                                               .
# Lance le serveur MkDocs en local
docs:
    uv run zensical serve --dev-addr localhost:8001

# Lance l'interface MLflow locale
mlflow:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# %% ML                                                                                .
train:
    uv run python -m credit_scoring.models.training_pipeline --run-cv

tune:
    uv run python -m credit_scoring.models.tune

export-onnx:
    uv run python scripts/export_onnx.py

# %% TEST                                                                              .

test:
    uv run pytest tests/ -v

lint:
    uv run ruff check src/

lint-ci:
	uv run --no-project ruff format src/ tests/
	uv run --no-project ruff check src/ tests/

# %% DOCKER                                                                            .

IMAGE_NAME := "credit-scoring"
CONTAINER_NAME := "credit-scoring-app"

# Build Docker image with cache
docker-build:
	docker build -t {{IMAGE_NAME}} .

# Build Docker image without cache
docker-build-clean:
	docker build --no-cache -t {{IMAGE_NAME}} .

# Run Docker image locally
docker-run:
	docker run --rm \
		--name {{CONTAINER_NAME}} \
		-p 8000:8000 \
		-p 7860:7860 \
		{{IMAGE_NAME}}

# Build and run local Docker image
docker-local: docker-build docker-run

# Stop running local container
docker-stop:
	-docker stop {{CONTAINER_NAME}}

# Remove local container if it exists
docker-rm:
	-docker stop {{CONTAINER_NAME}}
	-docker rm {{CONTAINER_NAME}}

# Remove local Docker image
docker-rmi:
	-docker rmi {{IMAGE_NAME}}

# Full local cleanup: container + image
docker-reset: docker-rm docker-rmi

# Rebuild from scratch and run
docker-fresh: docker-reset docker-build docker-run

# Rebuild from scratch without Docker cache and run
docker-fresh-no-cache: docker-reset docker-build-clean docker-run

# Build and run with Docker Compose
docker-up:
	docker compose up --build

# Stop Docker Compose stack
docker-down:
	docker compose down

# Stop Docker Compose stack and remove volumes/orphans
docker-clean:
	docker compose down --volumes --remove-orphans

# Full Compose cleanup including local image
docker-compose-reset: docker-clean docker-rmi