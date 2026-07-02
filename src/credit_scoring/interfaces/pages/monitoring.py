# src/credit_scoring/interfaces/pages/monitoring.py

# %% IMPORTS                                                                           .
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from plotly.subplots import make_subplots

# Paths from config
from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    EDUCATION_INVERSE,
    FILE_API,
    FILE_DRIFT_REPORT,
    FILE_PRED,
    FILE_QUALITY_REPORT,
    GENDER_INVERSE,
    NUMERICAL_FEATURES,
    PROD_REFERENCE,
    PROD_TEST,
)
from credit_scoring.serving.inference import get_model, get_reference_df

# %% IMPORTS                                                                           .
st.title("📡 Pilotage")


# Utilities
@st.cache_data(ttl=60)
def load_logs(file_path):
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_json(file_path, lines=True)


MONITORED_API_PATHS = {
    "/lookup/{sk_id}",
    "/model-info",
    "/predict",
    "/reference",
}


def normalize_api_path(path: str) -> str:
    normalized_path = path.rstrip("/") or "/"
    if normalized_path.startswith("/lookup/"):
        return "/lookup/{sk_id}"
    return normalized_path


# Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "🖥️ Observabilité API et modèle",
        "📊 Détection de dérive des données",
        "⚡ Optimisation de l’inférence modèle",
    ]
)


# %% TAB 1 : SRE Dashboard                                                      .
with tab1:
    st.header("État de l’API")
    api_df = load_logs(FILE_API)

    if not api_df.empty:
        api_df["path_group"] = api_df["path"].apply(normalize_api_path)
        api_df = api_df[api_df["path_group"].isin(MONITORED_API_PATHS)]
        api_df["timestamp"] = pd.to_datetime(api_df["timestamp"])

        if not api_df.empty:
            # Summary table
            st.subheader("Performance par route API")
            summary = (
                api_df.groupby("path_group")
                .agg(
                    {
                        "latency_ms": "mean",
                        "status_code": lambda x: (x >= 400).sum(),
                        "request_id": "count",
                    }
                )
                .rename(
                    columns={
                        "latency_ms": "Latence Moyenne (ms)",
                        "status_code": "Nb Erreurs",
                        "request_id": "Volume",
                    }
                )
            )
            st.dataframe(summary, use_container_width=True)

            st.divider()

            # Latency chart
            st.subheader("Analyse de la latence")

            # Compact selector
            col_sel1, _ = st.columns([1, 4])
            with col_sel1:
                selected_endpoint = st.selectbox(
                    "Route API à afficher:", sorted(api_df["path_group"].unique())
                )

            # Chart
            filtered_df = api_df[api_df["path_group"] == selected_endpoint]
            fig = px.line(
                filtered_df,
                x="timestamp",
                y="latency_ms",
                labels={"timestamp": "Date / Heure", "latency_ms": "Latence (ms)"},
                template="plotly_white",
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            pred_df = load_logs(FILE_PRED)
            required_cols = {
                "timestamp",
                "inference_ms",
                "cpu_percent",
                "memory_mb",
            }

            if not pred_df.empty and required_cols.issubset(pred_df.columns):
                if "success" in pred_df.columns:
                    pred_df = pred_df[pred_df["success"].fillna(False)]

                pred_df = pred_df.copy()
                pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
                for col in ["inference_ms", "cpu_percent", "memory_mb"]:
                    pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

                pred_df = pred_df.sort_values("timestamp")
                inference_df = pred_df.dropna(subset=["timestamp", "inference_ms"])
                runtime_df = pred_df.dropna(
                    subset=["timestamp", "cpu_percent", "memory_mb"],
                    how="any",
                )

                if not inference_df.empty:
                    st.subheader("Coûts runtime des prédictions")

                    fig_inference = px.line(
                        inference_df,
                        x="timestamp",
                        y="inference_ms",
                        labels={
                            "timestamp": "Date / Heure",
                            "inference_ms": "Temps d'inférence (ms)",
                        },
                        template="plotly_white",
                    )
                    fig_inference.update_layout(
                        height=320,
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig_inference, use_container_width=True)

                if not runtime_df.empty:
                    fig_runtime = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_runtime.add_trace(
                        go.Scatter(
                            x=runtime_df["timestamp"],
                            y=runtime_df["cpu_percent"],
                            name="CPU (%)",
                            mode="lines+markers",
                        ),
                        secondary_y=False,
                    )
                    fig_runtime.add_trace(
                        go.Scatter(
                            x=runtime_df["timestamp"],
                            y=runtime_df["memory_mb"],
                            name="Mémoire (MB)",
                            mode="lines+markers",
                        ),
                        secondary_y=True,
                    )
                    fig_runtime.update_layout(
                        height=320,
                        margin=dict(l=20, r=20, t=20, b=20),
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    fig_runtime.update_xaxes(title_text="Date / Heure")
                    fig_runtime.update_yaxes(title_text="CPU (%)", secondary_y=False)
                    fig_runtime.update_yaxes(
                        title_text="Mémoire (MB)",
                        secondary_y=True,
                    )
                    st.plotly_chart(fig_runtime, use_container_width=True)

        else:
            st.info("Aucune donnée de logs disponible pour les routes suivies.")

    else:
        st.info("Aucune donnée de logs disponible.")
# %% TAB 2 : Surveillance de la dérive des données

CAT_INVERSE_MAPS = {
    "CODE_GENDER": GENDER_INVERSE,
    "NAME_EDUCATION_TYPE": EDUCATION_INVERSE,
}


def prepare_monitoring_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for Evidently reports."""
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    ref_df = pd.read_parquet(PROD_REFERENCE)[features].copy()
    test_df = pd.read_parquet(PROD_TEST)[features].copy()

    # Check that no feature is used twice
    overlap = set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)
    if overlap:
        raise ValueError(f"Features used as both numerical and categorical: {overlap}")

    # Decode categorical features for readable reports
    for col, inverse_map in CAT_INVERSE_MAPS.items():
        ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce").map(inverse_map)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").map(inverse_map)

    # Force numerical columns
    for col in NUMERICAL_FEATURES:
        ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    # Force categorical columns
    for col in CATEGORICAL_FEATURES:
        ref_df[col] = ref_df[col].fillna("Inconnu").astype(str)
        test_df[col] = test_df[col].fillna("Inconnu").astype(str)

    return ref_df, test_df


def build_data_definition() -> DataDefinition:
    """Build Evidently data definition."""
    return DataDefinition(
        numerical_columns=NUMERICAL_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def build_evidently_datasets() -> tuple[Dataset, Dataset]:
    """Build Evidently Dataset objects with explicit column types."""
    ref_df, test_df = prepare_monitoring_data()
    data_definition = build_data_definition()

    ref_data = Dataset.from_pandas(
        ref_df,
        data_definition=data_definition,
    )

    test_data = Dataset.from_pandas(
        test_df,
        data_definition=data_definition,
    )

    return ref_data, test_data


def generate_drift_report() -> None:
    """Generate and save the data drift report."""
    ref_data, test_data = build_evidently_datasets()

    report = Report(
        [
            DataDriftPreset(
                num_method="wasserstein",
                cat_method="psi",
                cat_threshold=0.2,
            )
        ]
    )

    evaluation = report.run(
        current_data=test_data,
        reference_data=ref_data,
    )

    evaluation.save_html(str(FILE_DRIFT_REPORT))


def generate_quality_report() -> None:
    """Generate and save the data summary report."""
    ref_data, test_data = build_evidently_datasets()

    report = Report(
        [
            DataSummaryPreset(),
        ]
    )

    evaluation = report.run(
        current_data=test_data,
        reference_data=ref_data,
    )

    evaluation.save_html(str(FILE_QUALITY_REPORT))


def display_html_report(path: Path, height: int = 1000) -> None:
    """Display a saved HTML report."""
    with open(path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=height, scrolling=True)


with tab2:
    st.header("Surveillance de la dérive des données")

    col_1, col_2 = st.columns(2)

    with col_1:
        if st.button("Générer le rapport de dérive"):
            with st.spinner("Génération du rapport de dérive..."):
                try:
                    generate_drift_report()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport de dérive : {e}")

    with col_2:
        if st.button("Générer le rapport qualité"):
            with st.spinner("Génération du rapport qualité..."):
                try:
                    generate_quality_report()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport qualité : {e}")

    if FILE_DRIFT_REPORT.exists():
        st.success("Rapport de dérive disponible.")
        with st.expander("Afficher le rapport de dérive", expanded=True):
            display_html_report(FILE_DRIFT_REPORT, height=1000)

    if FILE_QUALITY_REPORT.exists():
        st.success("Rapport qualité disponible.")
        with st.expander("Afficher le rapport qualité"):
            display_html_report(FILE_QUALITY_REPORT, height=1000)


# %% TAB 3 : Optimisation de l’inférence modèle

with tab3:
    st.header("⚡ Optimisation de l’inférence modèle")

    st.markdown("""
    Analyse des performances d’inférence afin d’identifier les goulots d’étranglement
    et de justifier les optimisations du modèle en production.
    """)

    # Keep reports after Streamlit reruns
    if "profiling_report" not in st.session_state:
        st.session_state["profiling_report"] = None

    if "benchmark_report" not in st.session_state:
        st.session_state["benchmark_report"] = None

    col_run_1, col_run_2 = st.columns(2)

    with col_run_1:
        run_profiling = st.button("Lancer le profiling (50 inférences)")

    with col_run_2:
        run_benchmark = st.button("Lancer le benchmark LightGBM vs ONNX")

    if run_profiling:
        with st.spinner("Analyse des performances en cours..."):
            ref_df = get_reference_df()
            sample_features = ref_df.iloc[0].to_dict()

            def run_perf_test():
                """Run one model inference for profiling."""
                model, expected_features, _ = get_model()
                X = pd.DataFrame([sample_features])[expected_features]
                X = X.replace({None: np.nan}).astype(float)
                model.predict_proba(X)

            # Profile repeated inferences
            pr = cProfile.Profile()
            pr.enable()

            for _ in range(50):
                run_perf_test()

            pr.disable()

            # Extract profiling stats
            stats = pstats.Stats(pr)
            stats.strip_dirs()

            data = []
            for func_key, (cc, nc, tt, ct, _) in stats.stats.items():
                filename, line_number, func_name = func_key
                data.append(
                    {
                        "Fonction": f"{func_name} ({filename}:{line_number})",
                        "Nombre d’appels": nc,
                        "Temps propre (s)": tt,
                        "Temps cumulé (s)": ct,
                    }
                )

            df_profile = (
                pd.DataFrame(data)
                .sort_values(by="Temps propre (s)", ascending=False)
                .head(15)
            )

            st.session_state["profiling_report"] = df_profile

    if run_benchmark:
        with st.spinner("Benchmark en cours (100 inférences × 2)..."):
            import time

            from credit_scoring.serving.inference import (
                get_model,
                get_onnx_session,
                get_reference_df,
            )

            ref_df = get_reference_df()
            model, expected_features, _ = get_model()
            session = get_onnx_session()

            # Build input from already encoded reference data
            sample = ref_df.iloc[0]
            X_lgbm = (
                pd.DataFrame([sample])[expected_features]
                .replace({None: np.nan})
                .astype(float)
            )
            X_onnx = X_lgbm.astype(np.float32)

            input_name = session.get_inputs()[0].name
            N = 100

            # Warm-up to avoid first-call overhead
            model.predict_proba(X_lgbm)
            session.run(None, {input_name: X_onnx.values})

            # Benchmark LightGBM
            times_lgbm = []
            for _ in range(N):
                t0 = time.perf_counter()
                model.predict_proba(X_lgbm)
                times_lgbm.append((time.perf_counter() - t0) * 1000)

            # Benchmark ONNX Runtime
            times_onnx = []
            for _ in range(N):
                t0 = time.perf_counter()
                session.run(None, {input_name: X_onnx.values})
                times_onnx.append((time.perf_counter() - t0) * 1000)

            mean_lgbm = np.mean(times_lgbm)
            mean_onnx = np.mean(times_onnx)
            speedup = mean_lgbm / mean_onnx
            theoretical_rps = 1000 / mean_onnx

            results = pd.DataFrame(
                {
                    "Modèle": ["LightGBM standard", "LightGBM optimisé ONNX"],
                    "Latence moyenne (ms)": [mean_lgbm, mean_onnx],
                    "P95 (ms)": [
                        np.percentile(times_lgbm, 95),
                        np.percentile(times_onnx, 95),
                    ],
                    "P99 (ms)": [
                        np.percentile(times_lgbm, 99),
                        np.percentile(times_onnx, 99),
                    ],
                }
            )

            st.session_state["benchmark_report"] = {
                "results": results,
                "mean_lgbm": mean_lgbm,
                "mean_onnx": mean_onnx,
                "speedup": speedup,
                "theoretical_rps": theoretical_rps,
            }

    # Display both reports if available
    if st.session_state["profiling_report"] is not None:
        st.divider()
        st.subheader("Rapport de profiling")

        st.dataframe(
            st.session_state["profiling_report"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Temps propre (s)": st.column_config.NumberColumn(format="%.4f s"),
                "Temps cumulé (s)": st.column_config.NumberColumn(format="%.4f s"),
            },
        )

        st.info(
            "Ce tableau identifie les fonctions les plus coûteuses en temps CPU "
            "pendant les inférences."
        )

    if st.session_state["benchmark_report"] is not None:
        st.divider()
        st.subheader("Benchmark d’inférence : modèle standard vs ONNX")

        benchmark = st.session_state["benchmark_report"]
        results = benchmark["results"]
        mean_lgbm = benchmark["mean_lgbm"]
        mean_onnx = benchmark["mean_onnx"]
        speedup = benchmark["speedup"]
        theoretical_rps = benchmark["theoretical_rps"]

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Latence moyenne (ms)": st.column_config.NumberColumn(format="%.3f ms"),
                "P95 (ms)": st.column_config.NumberColumn(format="%.3f ms"),
                "P99 (ms)": st.column_config.NumberColumn(format="%.3f ms"),
            },
        )

        left, center, right = st.columns([1, 2, 1])

        with center:
            fig = px.bar(
                results,
                x="Modèle",
                y="Latence moyenne (ms)",
                text="Latence moyenne (ms)",
                title="Latence moyenne par modèle",
            )

            fig.update_traces(
                texttemplate="%{text:.3f} ms",
                textposition="outside",
            )

            fig.update_layout(
                height=320,
                width=520,
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=40),
                yaxis_title="Latence moyenne (ms)",
                xaxis_title=None,
            )

            st.plotly_chart(fig, use_container_width=False)

        st.success(f"🚀 ONNX est **{speedup:.1f}×** plus rapide en moyenne.")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latence moyenne LightGBM",
            f"{mean_lgbm:.2f} ms",
        )

        col2.metric(
            "Latence moyenne ONNX",
            f"{mean_onnx:.3f} ms",
            delta=f"-{mean_lgbm - mean_onnx:.2f} ms",
            delta_color="inverse",
        )

        col3.metric(
            "Accélération",
            f"{speedup:.1f}×",
            delta="vs LightGBM standard",
        )

        st.info(
            f"""
            **Analyse** : la version ONNX réduit fortement le temps d’inférence
            en exécutant un graphe optimisé via ONNX Runtime, avec moins de surcoût Python.

            À **{mean_onnx:.3f} ms par inférence**, l’ordre de grandeur théorique est
            d’environ **{theoretical_rps:,.0f} inférences/seconde** dans ce test local.
            """
        )
