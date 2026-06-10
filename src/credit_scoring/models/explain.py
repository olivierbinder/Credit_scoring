# src/credit_scoring/model/explain.py

import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from credit_scoring.logger import logger


def extract_and_plot_importance(
    model, feature_names: list, run_id: str, max_features: int = 25
):
    """
    Extracts global weights and logs to the currently active MLflow run.
    """
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

    # Plotting logic...
    df_top = df_imp.head(max_features).sort_values(
        by="absolute_importance", ascending=True
    )
    fig, ax = plt.subplots(figsize=(11, 8))

    if hasattr(model, "coef_") and (df_top["importance"] < 0).any():
        colors = ["#e74c3c" if val < 0 else "#2ecc71" for val in df_top["importance"]]
        ax.barh(df_top["feature"], df_top["importance"], color=colors)
        ax.axvline(0, color="#34495e", linestyle="--", alpha=0.7)
    else:
        ax.barh(df_top["feature"], df_top["absolute_importance"], color="#3498db")
    ax.set_xlabel(label)
    ax.set_title(f"Top {max_features} Features", fontsize=12, pad=15)

    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)

    # LOGGING DIRECTLY TO ACTIVE RUN
    csv_path = "plots/feature_importances.csv"
    df_imp.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="data")

    print("- Interpretability artifacts logged to MLflow.")


def compute_and_save_shap(
    model,
    X_df: pd.DataFrame,
    model_name: str,
    sample_size: int = 400,
    output_dir: str = "plots/shap_reports",
):
    """
    Computes SHAP values, generates the summary plot, and logs results
    directly to the active MLflow run.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare Data
    if len(X_df) > sample_size:
        X_sample = X_df.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_df

    model_type_str = type(model).__name__.lower()

    # 2. Compute SHAP Values
    try:
        if "logistic" in model_type_str or "linear" in model_type_str:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer(X_sample)
            shap_matrix = (
                shap_values.values[:, :, 1]
                if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3
                else (
                    shap_values.values
                    if hasattr(shap_values, "values")
                    else shap_values
                )
            )

        elif any(t in model_type_str for t in ["forest", "tree", "xgb", "lgb"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_matrix = (
                shap_values[1]
                if isinstance(shap_values, list)
                else (
                    shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
                )
            )

        else:
            background = shap.kmeans(X_sample, 10) if len(X_sample) > 10 else X_sample
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)
            shap_matrix = (
                shap_values[1] if isinstance(shap_values, list) else shap_values
            )

        # 3. Create Dataframe
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap})
            .sort_values(by="mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        # 4. Generate Plot
        plt.figure(figsize=(10, 6))
        top_15 = shap_df.head(15).iloc[::-1]
        plt.barh(
            top_15["feature"],
            top_15["mean_abs_shap"],
            color="darkslateblue",
            edgecolor="k",
            alpha=0.85,
        )
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title(f"Global Feature Importance Profiles (SHAP) — {model_name}")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"shap_profile_{model_name.lower()}.png")
        plt.savefig(plot_path, dpi=150)
        shap.summary_plot(shap_matrix, X_sample, show=False)
        mlflow.log_figure(plt.gcf(), "plots/shap_summary_plot.png")
        plt.close()

        # 5. LOG TO ACTIVE RUN (Fluent API)
        mlflow.log_artifact(plot_path, artifact_path="shap_reports")

        csv_path = os.path.join(output_dir, "shap_importances.csv")
        shap_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="shap_reports")

        print("- SHAP analysis logged to MLflow artifacts.")
        return shap_df

    except Exception as e:
        logger.error(f"❌ Critical error in SHAP analysis: {str(e)}")
        raise e
