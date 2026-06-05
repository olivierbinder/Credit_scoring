# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import logging
import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project
from credit_scoring.config import (
    DF_PROC_PATH,
    DIR_CONFIG,
    DIR_DATA_PROCESSED,
)
from credit_scoring.logger import logger
from credit_scoring.model.evaluate import (
    evaluate_and_log_metrics,
    generate_and_log_plots,
)
from credit_scoring.model.explain import (
    compute_and_save_shap,
    extract_and_plot_importance,
)
from credit_scoring.model.train import (
    optimize_threshold,
    run_cross_validation,
    train_production_model,
)
from credit_scoring.prepare.preprocess import (
    load_and_preprocess_all,
    select_important_features,
    select_top_variance_features,
)
from credit_scoring.utils import load_yaml_config

# SETTINGS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Suppress log pollution and framework warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.sklearn.autolog(disable=True)


# MAIN EXPERIMENT FUNCTION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def run_experiment(
    config, load_raw_data, run_cv, run_feat_imp, run_shap, max_rows=None, debug=False
):
    logger.info(msg="■■■■■ RUNNING EXPERIMENT ■■■■■")
    print(f"- YAML config: {config}")
    print(f"- Load raw data: {load_raw_data}")
    print(f"- Run Cros validation: {run_cv}")
    print(f"- Run Feature Importance: {run_feat_imp}")
    print(f"- Run SHAP: {run_shap}")
    print(f"- Debug: {debug}")

    cfg = load_yaml_config(config_path=DIR_CONFIG / config)

    # MLFlow config
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["experiment_name"])

    # Load and Process Data, or load processed data
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    if load_raw_data is False:
        logger.info(msg="■■■■■ Loading processed data and pipeline from cache ■■■■■")
        df = pd.read_parquet(DF_PROC_PATH)
    else:
        logger.info(msg="■■■■■ Loading raw data and preprocessing ■■■■■")
        num_rows = 10000 if debug else None
        df = load_and_preprocess_all(num_rows)

        logger.info(msg="■■■■■ Saving processed data ■■■■■")
        DIR_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DF_PROC_PATH, index=False)

    # Split Data
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    train_df = df[df["TARGET"].notnull()]

    if max_rows:
        train_df = train_df.sample(max_rows)

    X_train, y_train = (
        train_df.drop(columns=["TARGET", "SK_ID_CURR"]),
        train_df["TARGET"],
    )

    if "top_var" in cfg["preprocess"].lower():
        logger.info(msg="■■■■■ Filtering top variance features ■■■■■")
        X_train = select_top_variance_features(X_train)

    if "important_features" in cfg["preprocess"].lower():
        logger.info(msg="■■■■■ Filtering important features ■■■■■")
        X_train = select_important_features(X_train, "ft", 20)

    # Train Model
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    with mlflow.start_run(run_name=cfg["model"]) as run:
        logger.info("■■■■■ Starting MLFlow run ■■■■■")

        # 1. Log Metadata
        run_id = run.info.run_id
        hyperparams = cfg.get("hyperparameters", {})
        model_func_name = f"{cfg['model']}"
        base_name = f"{cfg['model']}_Preprocess-{cfg['preprocess']}"
        if max_rows:
            base_name += f"_rows-{max_rows}"
        run_name = f"{base_name}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_dict(cfg, "yaml")
        mlflow.log_param("model_type", cfg["model"])
        mlflow.log_param("executed_cv", run_cv)
        mlflow.log_metric("nb_features", X_train.shape[1])
        mlflow.log_metric("nb_rows", X_train.shape[0])

        if hyperparams:
            mlflow.log_params(hyperparams)

        # 2. Cross-Validation
        if run_cv:
            logger.info("■■■■■ Running Cross-Validation ■■■■■")
            cv_results = run_cross_validation(
                model_type=model_func_name, X=X_train, y=y_train, **hyperparams
            )
            for metric in ["roc_auc", "business_cost"]:
                key = f"test_{metric}"
                scores = np.array(cv_results[key])
                mlflow.log_metric(f"cv_{metric}_mean", np.mean(scores))
                mlflow.log_metric(f"cv_{metric}_std", np.std(scores))

        # 3. Training
        logger.info("■■■■■ Training model ■■■■■")
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        model = train_production_model(
            model_func_name,
            X_val_train,
            y_val_train,
            X_val_test,
            y_val_test,
            **hyperparams,
        )
        optimal_threshold = optimize_threshold(model, X_val_test, y_val_test)
        print(f"Seuil métier optimal : {optimal_threshold:.3f}")
        mlflow.log_param("evaluation_threshold", optimal_threshold)

        # Evaluate Model
        # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
        logger.info("■■■■■ Evaluating model ■■■■■")

        evaluate_and_log_metrics(
            model, X_val_train, y_val_train, "training", optimal_threshold
        )
        evaluate_and_log_metrics(
            model, X_val_test, y_val_test, "validation", optimal_threshold
        )
        generate_and_log_plots(
            model, X_val_test, y_val_test, "validation", optimal_threshold
        )

        feature_names = X_train.columns.tolist()
        mlflow.log_text("\n".join(feature_names), artifact_file="feature_names.txt")

        # Log model
        flavor_name = model.__class__.__module__.split(".")[0].lower()
        flavor = getattr(mlflow, flavor_name, mlflow.sklearn)
        signature = mlflow.models.infer_signature(
            X_val_test.head(5), model.predict(X_val_test.head(5))
        )
        flavor.log_model(model, artifact_path="model", signature=signature)

        # Explain Model
        # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
        if run_feat_imp:
            logger.info("■■■■■ Running Feature Importance ■■■■■")
            extract_and_plot_importance(model, feature_names, run_id=run_id)
        if run_shap:
            logger.info("■■■■■ Running SHAP ■■■■■")
            compute_and_save_shap(model, X_val_test, cfg["model"])
        logger.info("■■■■■ EXPERIMENT COMPLETED ■■■■■")

        # Generate
        # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
        """

        test_df = df[df["TARGET"].isnull()]
        X_test, y_test = (
            test_df.drop(columns=["TARGET", "SK_ID_CURR"]),
            test_df["TARGET"],
        )


        """
