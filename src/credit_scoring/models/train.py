# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Machine learning
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Project
from credit_scoring.logger import logger
from credit_scoring.utils import timer

MODEL_REGISTRY = {
    "dummy": DummyClassifier,
    "log_reg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
    "catboost": CatBoostClassifier,
}
SUPPORTS_EVAL_SET = {"lightgbm", "xgboost"}
SUPPORTS_SAMPLE_WEIGHT = {"lightgbm", "log_reg", "random_forest"}


# TRAIN FUNCTION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@timer
def train_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    **kwargs,
):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"❌ Model '{model_type}' unknown.")

    model = MODEL_REGISTRY[model_type](**kwargs)
    sw = np.where(y_train == 1, 10, 1)

    if model_type in ("lightgbm", "log_reg", "random_forest"):
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        # dummy, catboost (gère le déséquilibre via auto_class_weights dans kwargs)
        model.fit(X_train, y_train)
    return model


# CROSS VALIDATION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def business_cost_scorer(y_true, y_pred):
    """Calculate average business cost"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(((fn * 10) + (fp * 1)) / len(y_true))


def optimize_threshold(model, X_val, y_val):
    """Trouve le seuil qui minimise le business_cost sur le set de validation."""
    if not hasattr(model, "predict_proba"):
        return 0.5
    else:
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


@timer
def run_cross_validation(
    model_type: str, X: pd.DataFrame, y: pd.Series, rdstate: int, **kwargs
) -> dict:

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rdstate)

    cv_results = {"roc_auc": [], "business_cost": []}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. Entraînement robuste (avec early stopping et poids)
        model = train_model(model_type, X_train, y_train, **kwargs)

        # 2. Prédictions
        y_proba = model.predict_proba(X_val)[:, 1]

        # 3. Optimisation du seuil sur CE fold uniquement (très important !)
        best_thresh = optimize_threshold(model, X_val, y_val)
        y_pred = (y_proba >= best_thresh).astype(int)

        # 4. Calcul des scores
        cv_results["roc_auc"].append(roc_auc_score(y_val, y_proba))
        cv_results["business_cost"].append(business_cost_scorer(y_val, y_pred))

        logger.info(f"🆗 Fold {fold + 1} completed.")

    return cv_results
