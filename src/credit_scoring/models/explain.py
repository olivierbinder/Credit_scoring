# %%  IMPORTS                                                                          .
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from credit_scoring.logger import logger
from credit_scoring.utils import timer


# %%  FEATURE IMPORTANCE EXTRACTION                                                    .
@timer
def extract_and_plot_importance(model, feature_names: list, max_features: int = 25):
    """
    Extracts global weights and logs to the currently active MLflow run.
    """
    # Model-specific importances
    if hasattr(model, "coef_"):
        importances = model.coef_[0]
        label = "Coefficient Value (Directional)"
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Feature Importance Score (Gini/Gain)"
    else:
        logger.warning(
            f"❌ Model type {type(model).__name__} does not expose structural importances."
        )
        return

    # Log importances to MLflow
    df_imp = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "absolute_importance": np.abs(importances),
            }
        )
        .sort_values(by="absolute_importance", ascending=False)
        .reset_index(drop=True)
    )
    top_features = df_imp.head(max_features)["feature"].tolist()

    mlflow.log_table(data=df_imp, artifact_file="explain/feature_importance_all.json")
    mlflow.log_dict(
        {"top_features": top_features, "label": label},
        "explain/feature_importance_top.json",
    )

    # Generate and log feature importance plot
    df_plot = df_imp.head(max_features).sort_values("absolute_importance")

    fig, ax = plt.subplots(figsize=(8, max(4, max_features * 0.3)))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in df_plot["importance"]]
    ax.barh(df_plot["feature"], df_plot["importance"], color=colors)
    ax.set_xlabel(label)
    ax.set_title(f"Top {max_features} Feature Importances")
    fig.tight_layout()

    mlflow.log_figure(fig, "explain/feature_importance_plot.png")
    plt.close(fig)

    logger.info("🆗 Feature importance artifacts logged to MLflow.")
    return df_imp


# %%  SHAP VALUES EXTRACTION                                                           .
@timer
def extract_and_plot_shap(model, X_sample: pd.DataFrame, max_features: int = 25):
    """
    Computes SHAP values and logs to the currently active MLflow run.
    """
    # Compute SHAP values
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
    except Exception as e:
        logger.warning(
            f"❌ Could not compute SHAP values for {type(model).__name__}: {e}"
        )
        return
    if isinstance(shap_values, list):
        values = shap_values[0].values
    else:
        values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    # Log SHAP importances to MLflow
    mean_abs_shap = np.abs(values).mean(axis=0)

    df_shap = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_features = df_shap.head(max_features)["feature"].tolist()

    mlflow.log_table(data=df_shap, artifact_file="explain/shap_importance_all.json")
    mlflow.log_dict(
        {"top_shap_features": top_features},
        "explain/shap_importance_all_top.json",
    )

    # Generate and log SHAP beeswarm plot
    fig = plt.figure(figsize=(8, max(4, max_features * 0.3)))
    shap.summary_plot(values, X_sample, max_display=max_features, show=False)
    fig = plt.gcf()
    fig.tight_layout()

    mlflow.log_figure(fig, "explain/shap_importance_plot_beeswarm.png")
    plt.close(fig)

    # Generate and log SHAP importance bar plot
    df_plot = df_shap.head(max_features).sort_values("mean_abs_shap")

    fig2, ax = plt.subplots(figsize=(8, max(4, max_features * 0.3)))
    ax.barh(df_plot["feature"], df_plot["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {max_features} SHAP Feature Importances")
    fig2.tight_layout()

    mlflow.log_figure(fig2, "explain/shap_importance_plot_bar.png")
    plt.close(fig2)

    logger.info("🆗 SHAP importance artifacts logged to MLflow.")
    return df_shap
