from pathlib import Path

import mlflow

# Build the absolute path to the database
db_path = Path(__file__).resolve().parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
client = mlflow.tracking.MlflowClient()
for rm in client.search_registered_models():
    print(f"Model Name: {rm.name}")

print(f"Connecting to database at: {db_path}")
