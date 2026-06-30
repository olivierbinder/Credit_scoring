"""
Training pipeline module.

This module orchestrates the end-to-end model training workflow, from data
loading to model registration.
"""
# %%  IMPORTS                                                                          .

import argparse
import datetime
import gc
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Literal

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

# Project
from credit_scoring.config import (
    DIR_CONFIG,
    DIR_DATA,
    DIR_DATA_PROCESSED,
    FILE_DATA_PROCESSED,
    MLFLOW_TRACKING_URI,
)
from credit_scoring.features.preprocess import (
    load_and_preprocess_all,
    select_important_features,
)
from credit_scoring.logger import logger
from credit_scoring.models.evaluate import evaluate_and_log
from credit_scoring.models.explain import (
    extract_and_plot_importance,
    extract_and_plot_shap,
)
from credit_scoring.models.train import (
    optimize_threshold,
    run_cross_validation,
    train_model,
)

# %%  SETTINGS                                                                         .
# Silence framework warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.sklearn.autolog(disable=True)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 50)


# %%  YAML EXPERIMENT CONFIG PARSING                                                   .
class ExperimentConfig(BaseModel):
    name: str
    model: str
    data_loading: Literal["one", "two", "full"]
    ft_engineering: str
    ft_selection: str
    random_state: int
    test_size: float
    run: str
    model_params: Dict[str, Any]


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    exp_data = data["experiment"]
    model_name = exp_data["model"]
    params = data["models"][model_name]
    run_name = (
        f"{model_name.capitalize()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cfg = ExperimentConfig(**exp_data, run=run_name, model_params=params)

    return cfg


# %%  MAIN EXPERIMENT FUNCTION                                                         .
def run_experiment(
    config_path: Path = DIR_CONFIG / "training.yaml",
    load_raw_data=False,
    run_cv=False,
    run_feat_imp=False,
    run_shap=False,
    max_rows=None,
    kaggle=False,
):
    logger.info(msg="RUNNING EXPERIMENT")
    cfg = load_experiment_config(config_path)
    logger.info(f"ℹ️ Experiment name: {cfg.name}")
    logger.info(f"ℹ️ Run name: {cfg.run}")
    logger.info(f"ℹ️ Loading : {cfg.data_loading}")
    logger.info(f"ℹ️ Feature Engineering : {cfg.ft_engineering}")
    logger.info(f"ℹ️ Feature Selection : {cfg.ft_selection}")
    logger.info(f"ℹ️ Model: {cfg.model}\n")
    logger.info(f"ℹ️ Load from raw data: {load_raw_data}")
    logger.info(f"ℹ️ Run Cros validation: {run_cv}")
    logger.info(f"ℹ️ Run Feature Importance: {run_feat_imp}")
    logger.info(f"ℹ️ Run SHAP: {run_shap}")

    # Load processed data or preprocess raw sources.
    if load_raw_data is False:
        logger.info(msg="✅ Loading processed data from cache")
        df = pd.read_parquet(FILE_DATA_PROCESSED)
        logger.info(f"🆗 dataset loaded (shape = {df.shape[0]:,d} | {df.shape[1]:,d})")

    else:
        logger.info(msg="✅ Loading raw data and preprocessing")
        df = load_and_preprocess_all(max_rows)

        logger.info(msg="✅ Saving processed data")
        DIR_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FILE_DATA_PROCESSED, index=False)

    train_df = df[df["TARGET"].notnull()]
    test_df = df[df["TARGET"].isnull()]
    del df
    gc.collect()

    if max_rows:
        train_df = train_df.sample(max_rows)

    X, y = (
        train_df.drop(columns=["TARGET", "SK_ID_CURR"]),
        train_df["TARGET"],
    )
    del train_df
    gc.collect()

    # Feature selection.
    if "importan" in cfg.ft_selection.lower():
        logger.info(msg="✅ Filtering important features")
        X = select_important_features(X, 20)
        logger.info(
            f"🆗 Features selection : (shape = {X.shape[0]:,d} | {X.shape[1]:,d})"
        )

    # Split data.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    del X, y
    gc.collect()

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y_train_full,
    )
    del X_train_full, y_train_full
    gc.collect()

    # Experiment setup.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.name)
    with mlflow.start_run(run_name=cfg.run):
        logger.info("✅ Starting MLFlow run")

        # Logging.
        mlflow.set_tag("data_loading", cfg.data_loading)
        mlflow.set_tag("feature_engineering", cfg.ft_engineering)
        mlflow.set_tag("feature_selection", cfg.ft_selection)
        mlflow.set_tag("model", cfg.model)
        mlflow.set_tag("nb_features", X_train.shape[1])
        mlflow.set_tag("nb_rows_train", X_train.shape[0])
        mlflow.log_params(cfg.model_params)

        # Cross-validation.
        if run_cv:
            logger.info("✅ Running Cross-Validation")
            cv_results = run_cross_validation(
                model_type=cfg.model,
                X=X_train,
                y=y_train,
                rdstate=cfg.random_state,
                **cfg.model_params,
            )
            for metric in ["roc_auc", "business_cost"]:
                scores = np.array(cv_results[metric])
                mlflow.log_metric(f"cv_{metric}_mean", round(float(np.mean(scores)), 4))
                mlflow.log_metric(f"cv_{metric}_std", round(float(np.std(scores)), 4))

        # Training.
        logger.info("✅ Training model")

        model = train_model(
            cfg.model,
            X_train,
            y_train,
            **cfg.model_params,
        )
        logger.info("🆗 Production model trained")

        optimal_threshold = optimize_threshold(model, X_calib, y_calib)
        mlflow.log_param("evaluation_threshold", optimal_threshold)

        # Evaluation.
        logger.info("✅ Evaluating model")
        evaluate_and_log(model, X_test, y_test, "test", optimal_threshold)
        evaluate_and_log(model, X_calib, y_calib, "calibration", optimal_threshold)
        evaluate_and_log(model, X_train, y_train, "train", optimal_threshold)

        mlflow.set_tag("nb_rows_test", X_test.shape[0])

        # Explanation.
        feature_names = X_train.columns.tolist()
        mlflow.log_dict(feature_names, "features.json")

        if run_feat_imp:
            logger.info("✅ Running Feature Importance")
            extract_and_plot_importance(model, feature_names, max_features=25)
        if run_shap:
            logger.info("✅ Running SHAP")
            extract_and_plot_shap(
                model,
                X_test.sample(100, random_state=cfg.random_state),
                max_features=25,
            )

        # Logging.
        logger.info("✅ Logging model")
        artifact_name = (
            cfg.model
            + "_DATA_"
            + cfg.data_loading
            + "_ENG_"
            + cfg.ft_engineering
            + "_SEL_"
            + cfg.ft_selection
        )

        X_sample = X_test.sample(5)

        MODEL_LOGGERS = {
            "lightgbm": mlflow.lightgbm.log_model,
            "xgboost": mlflow.xgboost.log_model,
            "catboost": mlflow.catboost.log_model,
        }

        log_model = MODEL_LOGGERS.get(cfg.model, mlflow.sklearn.log_model)

        log_model(
            model,
            artifact_path=artifact_name,
            input_example=X_sample,
            metadata={"optimal_threshold": float(optimal_threshold)},
        )

        logger.info(f"🆗 Model '{cfg.model}' logged to MLflow.")

        # Generate Kaggle predictions.
        if kaggle:
            logger.info("✅ Generating Kaggle predictions")
            X_kaggle = test_df[feature_names]
            y_pred_kaggle = model.predict_proba(X_kaggle)[:, 1]
            y_pred_kaggle = pd.DataFrame(
                {"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_pred_kaggle}
            )
            y_pred_kaggle.to_csv(
                DIR_DATA / "Kaggle_predictions" / f"{cfg.run}.csv", index=False
            )
            logger.info("🆗 Kaggle predictions generated")


# %%  ARGUMENT PARSING                                                                 .
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run credit scoring training experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DIR_CONFIG / "training.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--load-raw-data", action="store_true", help="Reload raw data from source."
    )
    parser.add_argument("--run-cv", action="store_true", help="Run cross-validation.")
    parser.add_argument(
        "--run-feat-imp", action="store_true", help="Log feature importance artifacts."
    )
    parser.add_argument("--run-shap", action="store_true", help="Log SHAP artifacts.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows (for debugging).",
    )
    parser.add_argument(
        "--kaggle", action="store_true", help="Generate Kaggle predictions."
    )

    return parser.parse_args()


# %%  MAIN                                                                             .
if __name__ == "__main__":
    args = parse_args()

    run_experiment(
        config_path=args.config,
        load_raw_data=args.load_raw_data,
        run_cv=args.run_cv,
        run_feat_imp=args.run_feat_imp,
        run_shap=args.run_shap,
        max_rows=args.max_rows,
        kaggle=args.kaggle,
    )
