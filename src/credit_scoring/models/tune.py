# %%  IMPORTS                                                                          .
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

# Imports projet
from credit_scoring.config import FILE_DATA_PROCESSED
from credit_scoring.features.preprocess import select_important_features
from credit_scoring.models.train import business_cost_scorer

# %%  CONFIGURATION                                                                    .
MODEL_URI = "models:/LightGBM_20features/1"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Credit_Scoring_FineTuning"
FEATURES_TO_SELECT = 20


# %%  XXX                                                                  .
# 1. Chargement du modèle de base
print("Chargement du modèle depuis le registre...")
base_model = mlflow.lightgbm.load_model(MODEL_URI)

# 2. Préparation des données
print("Chargement et préparation des données...")
mlflow.set_tracking_uri(TRACKING_URI)
df = pd.read_parquet(FILE_DATA_PROCESSED)
train_df = df[df["TARGET"].notnull()]

X_train = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
y_train = train_df["TARGET"]

X_train = select_important_features(X_train, FEATURES_TO_SELECT)

# 3. Configuration de l'optimisation
scoring = {
    "roc_auc": "roc_auc",
    "business_cost": make_scorer(business_cost_scorer, greater_is_better=False),
}

param_dist = {
    "n_estimators": [500, 1000, 1500],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "reg_alpha": [0.01, 0.05],
    "reg_lambda": [0.05, 0.1],
}

# 4. Exécution du Fine-Tuning sous MLflow
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="Fine-tuning_RandomizedSearchCV"):
    print("Démarrage de la recherche d'hyperparamètres...")

    rs = RandomizedSearchCV(
        base_model,
        param_dist,
        refit="business_cost",  # Optimisation métier prioritaire
        scoring=scoring,
        cv=3,
        n_iter=6,
        verbose=2,
        n_jobs=-1,
    )
    rs.fit(X_train, y_train)

    print(f"Meilleurs paramètres : {rs.best_params_}")
    print(f"Meilleur score : {rs.best_score_}")

    # 5. Enregistrement du modèle optimisé
    signature = mlflow.models.infer_signature(
        X_train, rs.best_estimator_.predict(X_train)
    )

    mlflow.lightgbm.log_model(
        rs.best_estimator_,
        artifact_path="fine_tuned_model",
        signature=signature,
        registered_model_name="mon_modele_production",
    )

    # 6. Logging des métriques finales
    mlflow.log_params(rs.best_params_)

    metrics = {"best_business_cost": rs.best_score_}

    # Récupération sécurisée du score ROC AUC
    if "mean_test_roc_auc" in rs.cv_results_:
        metrics["best_roc_auc"] = rs.cv_results_["mean_test_roc_auc"][rs.best_index_]

    mlflow.log_metrics(metrics)
    print("Modèle fine-tuné et enregistré avec succès.")
