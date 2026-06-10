import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from credit_scoring.utils import timer


def compute_business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    """Calculate average business cost per sample (normalized)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return float(total_cost / len(y_true))  # Divided by total number of rows


@timer
def evaluate_and_log_metrics(model, X, y, dataset_name, threshold=0.5):
    """Compute metrics, log to MLflow with prefix, and return results."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(
        y, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )
    metrics = {
        f"{dataset_name}_roc_auc": roc_auc_score(y, y_proba),
        f"{dataset_name}_log_loss": log_loss(y, y_proba),
        f"{dataset_name}_recall_1": report["1"]["recall"],
        f"{dataset_name}_f1_1": report["1"]["f1-score"],
        f"{dataset_name}_business_cost": compute_business_cost(y, y_pred),
    }

    mlflow.log_metrics(metrics)
    mlflow.log_text(
        classification_report(y, y_pred), f"metrics/{dataset_name}_report.txt"
    )

    return metrics


@timer
def generate_and_log_plots(model, X, y, dataset_name, threshold=0.5):
    """Generate and log normalized Confusion Matrix and ROC Curve to MLflow."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

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

    print(f"- {dataset_name.capitalize()} plots logged to MLflow.")
