import shutil
from pathlib import Path

import mlflow

# Configuration
# Remplacez par le nom de votre modèle et la version souhaitée
model_uri = "models:/LightGBM_20features/1"
export_path = Path("model")


def export():
    # 1. Charger le modèle depuis MLflow
    print(f"Chargement du modèle depuis {model_uri}...")
    model = mlflow.lightgbm.load_model(model_uri)

    # 2. Créer le dossier local
    if export_path.exists():
        shutil.rmtree(export_path)
    export_path.mkdir()

    # 3. Sauvegarder localement
    # MLflow possède une méthode save_model pour exporter les artifacts
    print(f"Exportation du modèle vers {export_path}...")
    mlflow.lightgbm.save_model(model, path=str(export_path))
    print("Exportation réussie !")


if __name__ == "__main__":
    # Assurez-vous que votre URI de tracking pointe bien vers votre base locale
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    export()
