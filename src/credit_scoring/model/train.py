# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation

# Machine learning
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_scoring.config import MODEL_REGISTRY

# Project
from credit_scoring.utils import timer


def get_model_instance(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' unknown.")

    registry_entry = MODEL_REGISTRY[model_name]

    params = {**registry_entry["default_params"], **kwargs}  # ty:ignore[invalid-argument-type]

    return registry_entry["model_class"](**params)  # ty:ignore[call-non-callable]


def business_cost_scorer(y_true, y_pred):
    """Calculate average business cost"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(((fn * 10) + (fp * 1)) / len(y_true))


# TRAIN FUNCTION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@timer
def train_production_model(
    model_type: str, X_train: pd.DataFrame, y_train: np.ndarray, X_val, y_val, **kwargs
):
    """Unified single-fit training function replacing all specific functions."""
    model = get_model_instance(model_type, **kwargs)
    sw = np.where(y_train == 1, 10, 1) if "lightgbm" in model_type else None
    callbacks = [early_stopping(stopping_rounds=50), log_evaluation(period=100)]
    model.fit(
        X_train,
        y_train,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    return model


def optimize_threshold(model, X_val, y_val):
    """Trouve le seuil qui minimise le business_cost sur le set de validation."""
    y_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    min_cost = float("inf")

    # On teste 100 seuils pour trouver le meilleur
    for threshold in np.linspace(0.01, 0.99, 100):
        y_pred = (y_probs >= threshold).astype(int)
        cost = business_cost_scorer(y_val, y_pred)
        if cost < min_cost:
            min_cost = cost
            best_threshold = threshold

    return best_threshold


# CROSS VALIDATION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


@timer
def run_cross_validation(
    model_type: str, X: pd.DataFrame, y: pd.Series, **kwargs
) -> dict:

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = {"test_roc_auc": [], "test_business_cost": []}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. Entraînement robuste (avec early stopping et poids)
        model = train_production_model(
            model_type, X_train, y_train, X_val, y_val, **kwargs
        )

        # 2. Prédictions
        y_proba = model.predict_proba(X_val)[:, 1]

        # 3. Optimisation du seuil sur CE fold uniquement (très important !)
        best_thresh = optimize_threshold(model, X_val, y_val)
        y_pred = (y_proba >= best_thresh).astype(int)

        # 4. Calcul des scores
        cv_results["test_roc_auc"].append(roc_auc_score(y_val, y_proba))
        cv_results["test_business_cost"].append(business_cost_scorer(y_val, y_pred))

        print(f"- Fold {fold + 1} completed. Best threshold: {best_thresh:.3f}")

    return cv_results
