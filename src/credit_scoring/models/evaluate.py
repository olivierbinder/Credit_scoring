# src/credit_scoring/models/evaluate.py
# %%  IMPORTS                                                                          .
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_scoring.logger import logger
from credit_scoring.utils import timer


# %%  EVALUATION FUNCTIONS                                                             .
def compute_business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    """Calculate average business cost per sample (normalized)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return float(total_cost / len(y_true))  # Divided by total number of rows


@timer
def evaluate_and_log_metrics(y, y_proba, y_pred, dataset_name):
    """Compute metrics, log to MLflow with prefix, and return results."""
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    n = len(y)
    metrics = {
        # Primary / decision metrics
        f"{dataset_name}_business_cost": compute_business_cost(y, y_pred),
        f"{dataset_name}_roc_auc": roc_auc_score(y, y_proba),
        f"{dataset_name}_recall": recall_score(y, y_pred),
        f"{dataset_name}_fpr": fp / (fp + tn),  # Fallout Rate
        # Confusion matrix breakdown (ratios), grouped by predicted class
        f"{dataset_name}_tp": tp / n,
        f"{dataset_name}_fn": fn / n,
        f"{dataset_name}_tn": tn / n,
        f"{dataset_name}_fp": fp / n,
        # Summary
        f"{dataset_name}_positive_rate": y_pred.mean(),
        f"{dataset_name}_actual_positive_rate": y.mean(),
        # Secondary
        f"{dataset_name}_log_loss": log_loss(y, y_proba),
        f"{dataset_name}_precision": precision_score(y, y_pred),
        f"{dataset_name}_f1": f1_score(y, y_pred),
    }
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    mlflow.log_metrics(metrics)
    return metrics


@timer
def generate_and_log_plots(model, X, y, y_pred, dataset_name, threshold=0.5):
    """Generate and log normalized Confusion Matrix and ROC Curve to MLflow."""

    # 1. Normalized Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, ax=ax_cm, cmap="Blues", normalize="true"
    )
    ax_cm.set_title(
        f"Confusion Matrix ({dataset_name.capitalize()}) - Threshold: {threshold:.2f}"
    )
    mlflow.log_figure(fig_cm, f"plots/{dataset_name}_confusion_matrix.png")
    plt.close(fig_cm)

    # 2. ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax_roc)
    ax_roc.set_title(f"ROC Curve ({dataset_name.capitalize()})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    mlflow.log_figure(fig_roc, f"plots/{dataset_name}_roc_curve.png")
    plt.close(fig_roc)


# %%  MAIN FUNCTION                                                                    .
def evaluate_and_log(model, X, y, dataset_name, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_and_log_metrics(y, y_proba, y_pred, dataset_name)
    generate_and_log_plots(model, X, y, y_pred, dataset_name, threshold)
    logger.info(f"🆗 {dataset_name.capitalize()} metrics and plots logged to MLflow.")
    return metrics
