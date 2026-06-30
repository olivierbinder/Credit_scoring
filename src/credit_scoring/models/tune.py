# src/credit_scoring/models/tune.py

# %% IMPORTS
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from credit_scoring.config import DIR_CONFIG, FILE_DATA_PROCESSED, MLFLOW_TRACKING_URI
from credit_scoring.features.preprocess import select_important_features
from credit_scoring.models.evaluate import compute_business_cost
from credit_scoring.models.train import optimize_threshold
from credit_scoring.models.training_pipeline import load_experiment_config

TRACKING_URI = MLFLOW_TRACKING_URI
EXPERIMENT_NAME = "Credit_Scoring_FineTuning"
FEATURES_TO_SELECT = 20


def business_cost_with_threshold_search(estimator, X, y):
    """Custom scorer for RandomizedSearchCV using threshold optimization."""
    y_proba = estimator.predict_proba(X)[:, 1]

    best_cost = float("inf")

    for threshold in np.linspace(0.01, 0.99, 200):
        y_pred = (y_proba >= threshold).astype(int)
        cost = compute_business_cost(y, y_pred)

        if cost < best_cost:
            best_cost = cost

    return -best_cost


def run_tuning(config_path=DIR_CONFIG / "training.yaml"):
    cfg = load_experiment_config(config_path)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(disable=True)

    df = pd.read_parquet(FILE_DATA_PROCESSED)
    train_df = df[df["TARGET"].notnull()].copy()

    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = train_df["TARGET"]

    X = select_important_features(X, FEATURES_TO_SELECT)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y_train_full,
    )

    model_params = {
        **cfg.model_params,
        "n_jobs": 1,
    }

    model = LGBMClassifier(
        random_state=cfg.random_state,
        **model_params,
    )

    param_dist = {
        "n_estimators": [250, 300, 350],
        "learning_rate": [0.02, 0.03, 0.04],
        "reg_alpha": [0.9, 1.0, 1.1],
        "reg_lambda": [0.9, 1.0, 1.1],
    }

    with mlflow.start_run(run_name="RandomizedSearchCV_threshold_optimized"):
        print("Starting RandomizedSearchCV with threshold optimization...")

        mlflow.log_param("features_to_select", FEATURES_TO_SELECT)
        mlflow.log_params({f"base_{k}": v for k, v in cfg.model_params.items()})

        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring=business_cost_with_threshold_search,
            refit=True,
            cv=3,
            n_iter=8,
            n_jobs=-1,
            verbose=2,
            random_state=cfg.random_state,
        )

        rs.fit(X_train, y_train)
        best_model = rs.best_estimator_
        optimal_threshold = optimize_threshold(best_model, X_calib, y_calib)

        train_proba = best_model.predict_proba(X_train)[:, 1]
        train_pred = (train_proba >= optimal_threshold).astype(int)
        train_business_cost = compute_business_cost(y_train, train_pred)

        calib_proba = best_model.predict_proba(X_calib)[:, 1]
        calib_pred = (calib_proba >= optimal_threshold).astype(int)
        calib_business_cost = compute_business_cost(y_calib, calib_pred)

        test_proba = best_model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= optimal_threshold).astype(int)
        test_business_cost = compute_business_cost(y_test, test_pred)

        best_cv_business_cost = -rs.best_score_

        mlflow.log_params(rs.best_params_)
        mlflow.log_param("evaluation_threshold", float(optimal_threshold))

        mlflow.log_metric("best_cv_business_cost", float(best_cv_business_cost))
        mlflow.log_metric("train_business_cost", float(train_business_cost))
        mlflow.log_metric("calibration_business_cost", float(calib_business_cost))
        mlflow.log_metric("test_business_cost", float(test_business_cost))

        signature = mlflow.models.infer_signature(
            X_test,
            test_proba,
        )

        mlflow.lightgbm.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            metadata={"optimal_threshold": float(optimal_threshold)},
            registered_model_name="CreditScoring_LGBM_Tuned",
        )

        print(
            "Modèle optimisé enregistré. "
            f"CV: {best_cv_business_cost:.4f} | "
            f"Train: {train_business_cost:.4f} | "
            f"Calib: {calib_business_cost:.4f} | "
            f"Test: {test_business_cost:.4f} | "
            f"Seuil: {optimal_threshold:.4f}"
        )


if __name__ == "__main__":
    run_tuning()
