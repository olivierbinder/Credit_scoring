from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# credit_scoring/feature_selection/evaluate_feature_subsets.py
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_scoring.config import DF_PROC_PATH, DIR_CONFIG
from credit_scoring.models.evaluate import compute_business_cost
from credit_scoring.models.train import (
    optimize_threshold,
    train_model,
)
from credit_scoring.models.training_pipeline import load_experiment_config


def build_robust_feature_ranking(
    config_path: Path = DIR_CONFIG / "training.yaml",
    n_splits: int = 5,
    seeds: list[int] = [42, 123, 999],
):
    """
    Build a robust feature importance ranking using:
        - several CV folds
        - several random seeds

    Returns
    -------
    ranking_df : pd.DataFrame
    """

    cfg = load_experiment_config(config_path)
    print(f"ℹ️ Model: {cfg.model}\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    df = pd.read_parquet(DF_PROC_PATH)

    train_df = df[df["TARGET"].notnull()].copy()

    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = train_df["TARGET"]

    features = X.columns.tolist()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    all_importances = []

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    for seed in seeds:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            print(f"Seed={seed} | Fold={fold + 1}/{n_splits}")

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            params = cfg.model_params.copy()

            model = train_model(
                cfg.model,
                X_train,
                y_train,
                **params,
            )

            fold_importance = pd.DataFrame(
                {
                    "feature": features,
                    "importance": model.feature_importances_,
                    "seed": seed,
                    "fold": fold,
                }
            )

            all_importances.append(fold_importance)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    imp_df = pd.concat(all_importances, ignore_index=True)

    ranking_df = (
        imp_df.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            std_importance=("importance", "std"),
            max_importance=("importance", "max"),
            min_importance=("importance", "min"),
        )
        .reset_index()
    )

    # Nombre de runs où la feature est utilisée
    ranking_df["presence_rate"] = (
        imp_df.groupby("feature")["importance"].apply(lambda x: (x > 0).mean()).values
    )

    ranking_df = ranking_df.sort_values(
        "mean_importance",
        ascending=False,
    )

    ranking_df["rank"] = np.arange(
        1,
        len(ranking_df) + 1,
    )

    return ranking_df


def evaluate_feature_subsets(
    ranking_path: Path,
    min_features: int = 3,
    max_features: int = 25,
):
    """
    Evaluate model performance while progressively adding features
    according to the robust feature ranking.

    Returns
    -------
    pd.DataFrame
    """

    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------

    cfg = load_experiment_config(DIR_CONFIG / "training.yaml")

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    df = pd.read_parquet(DF_PROC_PATH)

    train_df = df[df["TARGET"].notnull()].copy()

    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])

    y = train_df["TARGET"]

    # ------------------------------------------------------------
    # Train / Test split
    # ------------------------------------------------------------

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    ranking = pd.read_csv(ranking_path)

    ordered_features = ranking["feature"].tolist()

    # ------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------

    results = []

    for k in range(
        min_features,
        max_features + 1,
    ):
        selected_features = ordered_features[:k]

        X_train = X_train_full[selected_features]
        X_test = X_test_full[selected_features]

        print(f"Training with {k} features...")

        model = train_model(
            cfg.model,
            X_train,
            y_train,
            **cfg.model_params,
        )

        threshold = optimize_threshold(
            model,
            X_test,
            y_test,
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "business_cost": compute_business_cost(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        results.append(
            {
                "n_features": k,
                "roc_auc": metrics["roc_auc"],
                "business_cost": metrics["business_cost"],
                "threshold": threshold,
                "features": selected_features,
            }
        )

    results_df = pd.DataFrame(results)

    return results_df
