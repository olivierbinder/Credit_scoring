# src/credit_scoring/serving/drift_analysis.py
"""
drift_analysis.py
=================
Automatic analysis of production logs for the Credit Scoring API.

Detects:
  1. Data drift  — feature distribution shift vs. reference (Evidently AI)
  2. Prediction drift — output probability / label shift
  3. Operational anomalies — error rate, latency spikes

Usage
-----
    python drift_analysis.py \
        --predictions logs/predictions.jsonl \
        --api-calls   logs/api_calls.jsonl   \
        --reference   data/reference.parquet \
        --output      reports/

"""

# %%  IMPORTS                                                                          .
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from credit_scoring.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

console = Console()

# %%  CONSTANTS                                                                        .
# Operational thresholds
ERROR_RATE_THRESHOLD = 0.05  # alert if > 5 % of calls fail
P95_LATENCY_THRESHOLD_MS = 2000  # alert if P95 latency > 2 s
DRIFT_SHARE_THRESHOLD = 0.3  # alert if > 30 % of features drift


# %%  DATA LOADING                                                                     .
def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_predictions(path: Path) -> pd.DataFrame:
    df = load_jsonl(path)
    # Flatten nested `inputs` dict into top-level columns
    inputs_df = pd.json_normalize(df["inputs"])
    df = pd.concat([df.drop(columns=["inputs"]), inputs_df], axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_reference(path: Path, sample: int | None = 5000) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Keep only the feature columns we track
    cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    df = df[cols]
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df


# %%  OPERATIONAL ANOMALIES                                                            .
def analyze_operations(api_df: pd.DataFrame) -> dict:
    """
    Compute key operational KPIs and flag anomalies.
    Returns a dict of metrics + a list of alerts.
    """
    total = len(api_df)
    errors = api_df["is_error"].sum()
    error_rate = errors / total if total else 0.0

    latency = api_df["latency_ms"].dropna()
    p50 = float(latency.quantile(0.50)) if len(latency) else 0.0
    p95 = float(latency.quantile(0.95)) if len(latency) else 0.0
    p99 = float(latency.quantile(0.99)) if len(latency) else 0.0

    alerts = []
    if error_rate > ERROR_RATE_THRESHOLD:
        alerts.append(
            f"🔴 Error rate {error_rate:.1%} exceeds threshold {ERROR_RATE_THRESHOLD:.0%}"
        )
    if p95 > P95_LATENCY_THRESHOLD_MS:
        alerts.append(
            f"🔴 P95 latency {p95:.0f} ms exceeds threshold {P95_LATENCY_THRESHOLD_MS} ms"
        )

    # Per-endpoint breakdown
    by_path = (
        api_df.groupby("path")
        .agg(
            calls=("status_code", "count"),
            errors=("is_error", "sum"),
            p50_ms=("latency_ms", lambda x: round(x.quantile(0.50), 1)),
            p95_ms=("latency_ms", lambda x: round(x.quantile(0.95), 1)),
        )
        .reset_index()
    )
    by_path["error_rate"] = (by_path["errors"] / by_path["calls"]).round(4)

    return {
        "total_calls": total,
        "error_count": int(errors),
        "error_rate": round(error_rate, 4),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "latency_p99_ms": round(p99, 1),
        "alerts": alerts,
        "by_endpoint": by_path.to_dict(orient="records"),
    }


# %%  PREDICTION DRIFT                                                                 .
def analyze_prediction_drift(pred_df: pd.DataFrame) -> dict:
    """
    Compare probability distribution over time using a rolling window.
    Flags if the mean probability shifts substantially (>0.1) from a
    stable baseline (first 20 % of records).
    """
    successful = pred_df[pred_df["success"]].copy()
    if successful.empty:
        return {"alert": "No successful predictions to analyse."}

    successful = successful.sort_values("timestamp")
    n = len(successful)
    baseline_end = max(1, int(n * 0.20))

    baseline_proba = successful.iloc[:baseline_end]["probability"]
    recent_proba = successful.iloc[baseline_end:]["probability"]

    baseline_mean = float(baseline_proba.mean())
    recent_mean = float(recent_proba.mean()) if len(recent_proba) else baseline_mean
    delta = abs(recent_mean - baseline_mean)

    default_rate_baseline = float((baseline_proba >= 0.5).mean())
    default_rate_recent = (
        float((recent_proba >= 0.5).mean())
        if len(recent_proba)
        else default_rate_baseline
    )

    alerts = []
    if delta > 0.10:
        alerts.append(
            f"🔴 Mean probability shifted by {delta:.3f} "
            f"(baseline={baseline_mean:.3f} → recent={recent_mean:.3f})"
        )

    return {
        "n_predictions": n,
        "baseline_size": baseline_end,
        "baseline_mean_probability": round(baseline_mean, 4),
        "recent_mean_probability": round(recent_mean, 4),
        "probability_delta": round(delta, 4),
        "baseline_default_rate": round(default_rate_baseline, 4),
        "recent_default_rate": round(default_rate_recent, 4),
        "alerts": alerts,
    }


# %%  DATA DRIFT — Evidently AI                                                        .
def analyze_data_drift(
    pred_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Run Evidently data drift report comparing production inputs to reference.
    Saves an HTML report to output_dir.
    """
    successful = pred_df[pred_df["success"]].copy()

    # Keep only feature columns present in both datasets
    num_cols = [
        c
        for c in NUMERICAL_FEATURES
        if c in successful.columns and c in reference_df.columns
    ]
    cat_cols = [
        c
        for c in CATEGORICAL_FEATURES
        if c in successful.columns and c in reference_df.columns
    ]
    all_cols = num_cols + cat_cols

    current_df = successful[all_cols].copy()
    ref_df = reference_df[all_cols].copy()

    # Cast categoricals to string so Evidently treats them correctly
    for col in cat_cols:
        current_df[col] = current_df[col].astype(str)
        ref_df[col] = ref_df[col].astype(str)

    column_mapping = ColumnMapping(
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            *[ColumnDriftMetric(column_name=col) for col in all_cols],
        ]
    )
    report.run(
        reference_data=ref_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))

    # Extract summary from report dict
    result = report.as_dict()
    metrics = result.get("metrics", [])

    dataset_drift_metric = next(
        (m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {}
    )
    drift_result = dataset_drift_metric.get("result", {})
    drift_share = drift_result.get("share_of_drifted_columns", 0.0)
    n_drifted = drift_result.get("number_of_drifted_columns", 0)
    dataset_drifted = drift_result.get("dataset_drift", False)

    # Per-column drift summary
    drifted_features = []
    for m in metrics:
        if m.get("metric") == "ColumnDriftMetric":
            col_result = m.get("result", {})
            if col_result.get("drift_detected"):
                drifted_features.append(
                    {
                        "feature": col_result.get("column_name"),
                        "stattest": col_result.get("stattest_name"),
                        "p_value": col_result.get("p_value"),
                        "threshold": col_result.get("stattest_threshold"),
                    }
                )

    alerts = []
    if drift_share > DRIFT_SHARE_THRESHOLD:
        alerts.append(
            f"🔴 Data drift detected on {drift_share:.0%} of features "
            f"({n_drifted} / {len(all_cols)} columns)"
        )
    for f in drifted_features:
        p_value = f.get("p_value")
        p_value_str = f"{p_value:.4f}" if p_value is not None else "?"

        alerts.append(
            f"⚠️  Feature '{f['feature']}' drifted "
            f"(test={f['stattest']}, p={p_value_str})"
        )

    return {
        "report_path": str(html_path),
        "dataset_drift_detected": dataset_drifted,
        "drift_share": round(drift_share, 4),
        "n_drifted_features": n_drifted,
        "drifted_features": drifted_features,
        "alerts": alerts,
    }


# %%  MISSING VALUES ANALYSIS                                                          .
def analyze_missing_values(pred_df: pd.DataFrame) -> dict:
    """
    Check if nullable features have a suspiciously high missing rate in production
    compared to expected nullable fields.
    """
    successful = pred_df[pred_df["success"]].copy()
    nullable = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_3",
        "DAYS_EMPLOYED",
        "OWN_CAR_AGE",
        "ACTIVE_DAYS_CREDIT_MAX",
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
        "CC_CNT_DRAWINGS_CURRENT_VAR",
    ]
    cols = [c for c in nullable if c in successful.columns]
    missing_rates = {}
    alerts = []
    for col in cols:
        rate = float(successful[col].isna().mean())
        missing_rates[col] = round(rate, 4)
        if rate > 0.80:
            alerts.append(f"⚠️  '{col}' has {rate:.0%} missing values in production")

    return {"missing_rates": missing_rates, "alerts": alerts}


def analyze_performance(pred_df: pd.DataFrame) -> dict:
    """
    Analyze inference performance metrics.
    """

    successful = pred_df[pred_df["success"]].copy()

    if successful.empty:
        return {"alerts": ["No successful predictions"]}

    inference = successful["inference_ms"].dropna()

    p50 = float(inference.quantile(0.50))
    p95 = float(inference.quantile(0.95))
    p99 = float(inference.quantile(0.99))

    cpu_avg = float(successful["cpu_percent"].mean())
    mem_avg = float(successful["memory_mb"].mean())

    alerts = []

    if p95 > 500:
        alerts.append(f"⚠️ Inference P95 exceeds target (P95={p95:.0f} ms)")

    return {
        "inference_p50_ms": round(p50, 2),
        "inference_p95_ms": round(p95, 2),
        "inference_p99_ms": round(p99, 2),
        "cpu_avg_percent": round(cpu_avg, 2),
        "memory_avg_mb": round(mem_avg, 2),
        "alerts": alerts,
    }


# %%  REPORTING — Rich console output                                                  .
def print_section(title: str, data: dict, console: Console):
    alerts = data.get("alerts", [])
    table = Table(show_header=False, box=None, padding=(0, 1))
    for key, value in data.items():
        if key in ("alerts", "by_endpoint", "drifted_features", "missing_rates"):
            continue
        table.add_row(f"[dim]{key}[/dim]", f"[bold]{value}[/bold]")

    color = "red" if alerts else "green"
    console.print(Panel(table, title=f"[bold]{title}[/bold]", border_style=color))

    if alerts:
        for alert in alerts:
            console.print(f"  {alert}")
    else:
        console.print("  [green]✅ No anomalies detected[/green]")


def save_summary(results: dict, output_dir: Path):
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n📄 Summary saved → [cyan]{summary_path}[/cyan]")


# %%  MAIN                                                                             .
def main():
    parser = argparse.ArgumentParser(
        description="Credit Scoring API — drift & anomaly analysis"
    )
    parser.add_argument(
        "--predictions",
        default="logs/predictions.jsonl",
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--api-calls", default="logs/api_calls.jsonl", help="Path to api_calls.jsonl"
    )
    parser.add_argument(
        "--reference",
        default="data/processed/reference.parquet",
        help="Path to reference parquet",
    )
    parser.add_argument(
        "--output", default="reports/", help="Output directory for reports"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule(
        "[bold blue]Credit Scoring — Production Monitoring Analysis[/bold blue]"
    )
    console.print(f"🕒 Run at: {datetime.now(timezone.utc).isoformat()}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    pred_path = Path(args.predictions)
    api_path = Path(args.api_calls)
    ref_path = Path(args.reference)

    if not pred_path.exists():
        console.print(f"[red]❌ Predictions log not found: {pred_path}[/red]")
        sys.exit(1)

    pred_df = load_predictions(pred_path)
    console.print(f"✔ Loaded {len(pred_df):,} prediction records")

    api_df = None
    if api_path.exists():
        api_df = load_jsonl(api_path)
        api_df["timestamp"] = pd.to_datetime(api_df["timestamp"], utc=True)
        console.print(f"✔ Loaded {len(api_df):,} API call records")

    reference_df = None
    if ref_path.exists():
        reference_df = load_reference(ref_path)
        console.print(f"✔ Loaded {len(reference_df):,} reference samples\n")

    all_results = {"run_timestamp": datetime.now(timezone.utc).isoformat()}

    # ── 1. Operational anomalies ───────────────────────────────────────────────
    console.rule("1 · Operational Health")
    if api_df is not None:
        ops = analyze_operations(api_df)
        print_section("API Operations", ops, console)
        all_results["operations"] = ops
    else:
        console.print(
            "[yellow]⚠️  No api_calls.jsonl — skipping operational analysis[/yellow]"
        )

    # ── 2. Prediction drift ────────────────────────────────────────────────────
    console.rule("2 · Prediction Drift")
    pred_drift = analyze_prediction_drift(pred_df)
    print_section("Prediction Drift", pred_drift, console)
    all_results["prediction_drift"] = pred_drift

    # ── 3. Data drift (Evidently) ─────────────────────────────────────────────
    console.rule("3 · Feature Data Drift (Evidently AI)")
    if reference_df is not None:
        data_drift = analyze_data_drift(pred_df, reference_df, output_dir)
        print_section("Data Drift", data_drift, console)
        all_results["data_drift"] = data_drift
        console.print(
            f"\n  📊 Full HTML report → [cyan]{data_drift['report_path']}[/cyan]"
        )
    else:
        console.print(
            "[yellow]⚠️  Reference parquet not found — skipping Evidently drift analysis[/yellow]"
        )

    # ── 4. Missing value analysis ─────────────────────────────────────────────
    console.rule("4 · Missing Value Analysis")
    missing = analyze_missing_values(pred_df)
    print_section("Missing Values", missing, console)
    all_results["missing_values"] = missing

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule("Summary")
    all_alerts = []
    for section_key in (
        "operations",
        "prediction_drift",
        "data_drift",
        "missing_values",
    ):
        section = all_results.get(section_key, {})
        all_alerts.extend(section.get("alerts", []))

    if all_alerts:
        console.print(f"[bold red]⚠️  {len(all_alerts)} alert(s) raised:[/bold red]")
        for alert in all_alerts:
            console.print(f"  {alert}")
    else:
        console.print(
            "[bold green]✅ All checks passed — no anomalies detected.[/bold green]"
        )

    save_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
