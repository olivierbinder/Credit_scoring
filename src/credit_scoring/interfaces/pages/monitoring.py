# src/credit_scoring/interfaces/pages/monitoring.py
"""
Page « Monitoring » — drift, anomalies opérationnelles, exploration des logs.

Accès via la navigation principale de l'app (📡 Monitoring).
Les chemins de logs sont configurables dans la sidebar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from credit_scoring.interfaces.monitoring.data import (
    DIR_API,
    DIR_PRED,
    DIR_REFERENCE,
    NULLABLE_FEATURES,
    NUMERICAL_FEATURES,
    load_api_calls,
    load_predictions,
    load_reference,
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — configuration des chemins + filtre temporel
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    pred_path = st.text_input("Prediction log", value=DIR_PRED)
    api_path = st.text_input("API call log", value=DIR_API)
    ref_path = st.text_input("Reference data", value=DIR_REFERENCE)

    if st.button("🔄 Recharger les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Filtre temporel")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
pred_df = load_predictions(pred_path)
api_df = load_api_calls(api_path)
ref_df = load_reference(ref_path)

st.title("📡 Monitoring de production")

if pred_df is None:
    st.warning(
        f"Aucun log de prédiction trouvé à `{pred_path}`. "
        "Lancez l'API avec le `api.py` mis à jour pour commencer à collecter des logs."
    )
    st.stop()

successful = pred_df[pred_df["success"]].copy()

# ── Filtre temporel (sidebar, après chargement) ────────────────────────────────
with st.sidebar:
    min_ts = pred_df["timestamp"].min().date()
    max_ts = pred_df["timestamp"].max().date()
    date_range = st.date_input(
        "Période",
        value=(min_ts, max_ts),
        min_value=min_ts,
        max_value=max_ts,
    )

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    mask = (pred_df["timestamp"].dt.date >= start_date) & (
        pred_df["timestamp"].dt.date <= end_date
    )
    pred_df_f = pred_df[mask]
    successful_f = pred_df_f[pred_df_f["success"]]
else:
    pred_df_f = pred_df
    successful_f = successful

# ── Bandeau de statut global ──────────────────────────────────────────────────
n_preds = len(pred_df_f)
n_ok = int(pred_df_f["success"].sum())
err_rate = 1 - n_ok / n_preds if n_preds else 0.0

baseline_end = max(1, int(len(successful) * 0.20))
baseline_mean = (
    float(successful.sort_values("timestamp").iloc[:baseline_end]["probability"].mean())
    if not successful.empty
    else 0.0
)
recent_mean = (
    float(successful_f["probability"].mean()) if not successful_f.empty else 0.0
)
prob_delta = abs(recent_mean - baseline_mean)

has_op_alert = err_rate > 0.05
has_drift_alert = prob_delta > 0.10

alert_parts = []
if has_op_alert:
    alert_parts.append(f"Taux d'erreur élevé ({err_rate:.1%})")
if has_drift_alert:
    alert_parts.append(f"Drift de prédiction détecté (Δ={prob_delta:+.3f})")

if alert_parts:
    st.error("🔴 " + " · ".join(alert_parts))
else:
    st.success("✅ Aucune anomalie détectée sur la période sélectionnée")

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_ops, tab_pred, tab_feat, tab_missing, tab_logs, tab_perf = st.tabs(
    [
        "🏥 Opérations",
        "🎯 Prédictions",
        "📈 Feature Drift",
        "❓ Valeurs manquantes",
        "⚡ Performance",
        "🗂️ Logs",
    ]
)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPERATIONAL HEALTH
# ════════════════════════════════════════════════════════════════════════════════
with tab_ops:
    st.subheader("Santé opérationnelle")

    if api_df is None:
        st.info(
            "Aucun fichier `api_calls.jsonl` trouvé. "
            "Démarrez l'API avec le `api.py` mis à jour."
        )
    else:
        api_f = (
            api_df[
                (api_df["timestamp"].dt.date >= date_range[0])
                & (api_df["timestamp"].dt.date <= date_range[-1])
            ]
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2
            else api_df
        )

        total = len(api_f)
        errors = int(api_f["is_error"].sum())
        error_rate = errors / total if total else 0.0
        lat = api_f["latency_ms"].dropna()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Appels totaux", f"{total:,}")
        c2.metric("Erreurs", f"{errors:,}")
        c3.metric(
            "Taux d'erreur",
            f"{error_rate:.1%}",
            delta="⚠️ ÉLEVÉ" if error_rate > 0.05 else "OK",
            delta_color="inverse" if error_rate > 0.05 else "normal",
        )
        c4.metric("Latence P50", f"{lat.quantile(0.50):.0f} ms")
        c5.metric(
            "Latence P95",
            f"{lat.quantile(0.95):.0f} ms",
            delta="⚠️ ÉLEVÉ" if lat.quantile(0.95) > 2000 else "OK",
            delta_color="inverse" if lat.quantile(0.95) > 2000 else "normal",
        )

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Volume d'appels (par heure)**")
            vol = api_f.set_index("timestamp").resample("1h")["status_code"].count()
            st.line_chart(vol.rename("appels/h"))
        with col_b:
            st.markdown("**Latence P50 / P95 (par heure)**")
            lat_ts = (
                api_f.set_index("timestamp")["latency_ms"]
                .resample("1h")
                .agg(P50=lambda x: x.quantile(0.50), P95=lambda x: x.quantile(0.95))
            )
            st.line_chart(lat_ts)

        st.divider()
        st.markdown("**Répartition par endpoint**")
        by_path = (
            api_f.groupby("path")
            .agg(
                Appels=("status_code", "count"),
                Erreurs=("is_error", "sum"),
                P50_ms=("latency_ms", lambda x: round(x.quantile(0.50), 1)),
                P95_ms=("latency_ms", lambda x: round(x.quantile(0.95), 1)),
            )
            .reset_index()
        )
        by_path["Taux d'erreur"] = (by_path["Erreurs"] / by_path["Appels"]).map(
            "{:.1%}".format
        )
        st.dataframe(by_path, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTION DRIFT
# ════════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Suivi des prédictions")

    if successful_f.empty:
        st.warning("Aucune prédiction réussie sur la période sélectionnée.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prédictions", f"{len(successful_f):,}")
        c2.metric("P(défaut) baseline", f"{baseline_mean:.3f}")
        c3.metric(
            "P(défaut) récente",
            f"{recent_mean:.3f}",
            delta=f"{recent_mean - baseline_mean:+.3f}",
            delta_color="inverse" if prob_delta > 0.10 else "normal",
        )

        if prob_delta > 0.10:
            st.error(
                f"🔴 Drift détecté — probabilité moyenne décalée de {prob_delta:.3f}"
            )

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Distribution des probabilités (production)**")
            hist = (
                successful_f["probability"]
                .pipe(lambda s: pd.cut(s, bins=20))
                .value_counts()
                .sort_index()
                .reset_index()
            )

            hist.columns = ["probability_range", "count"]
            hist["probability_range"] = hist["probability_range"].astype(str)

            st.bar_chart(
                hist,
                x="probability_range",
                y="count",
            )
        with col_b:
            st.markdown("**Probabilité moyenne dans le temps**")
            prob_ts = (
                successful_f.set_index("timestamp")["probability"].resample("1h").mean()
            )
            st.line_chart(prob_ts.rename("P(défaut) moyen"))

        st.markdown("**Taux de défaut prédit (par heure)**")
        dr_ts = (
            successful_f.set_index("timestamp")["probability"]
            .resample("1h")
            .apply(lambda x: (x >= 0.5).mean())
        )
        st.line_chart(dr_ts.rename("taux de défaut"))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE DRIFT
# ════════════════════════════════════════════════════════════════════════════════
with tab_feat:
    st.subheader("Drift des features")

    if ref_df is None:
        st.info(f"Données de référence introuvables (`{ref_path}`).")
    elif successful_f.empty:
        st.warning("Aucune donnée de production à comparer.")
    else:
        # Rapport Evidently pré-généré
        drift_report_path = Path("reports/drift_report.html")
        if drift_report_path.exists():
            st.markdown("**Rapport Evidently AI**")
            with open(drift_report_path, encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=820, scrolling=True)
            st.caption(
                "Rapport généré par `drift_analysis.py`. "
                "Relancez le script pour le mettre à jour."
            )
        else:
            st.info(
                "Aucun rapport Evidently trouvé. "
                "Lancez `python drift_analysis.py` pour en générer un, "
                "puis rechargez la page."
            )

        st.divider()
        st.markdown("**Comparaison manuelle production vs. référence**")

        num_cols_available = [
            c
            for c in NUMERICAL_FEATURES
            if c in successful_f.columns and c in ref_df.columns
        ]
        if num_cols_available:
            selected = st.selectbox("Feature", num_cols_available)

            prod_vals = successful_f[selected].dropna()
            ref_vals = ref_df[selected].dropna()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Production** (n={len(prod_vals):,})")
                hist = pd.cut(prod_vals, bins=30).value_counts().sort_index()

                hist.index = hist.index.astype(str)

                st.bar_chart(hist)

            with col_b:
                st.markdown(f"**Référence** (n={len(ref_vals):,})")
                hist = pd.cut(ref_vals, bins=30).value_counts().sort_index()

                hist.index = hist.index.astype(str)

                st.bar_chart(hist)

            stat_cols = st.columns(4)
            stat_cols[0].metric("Moy. prod", f"{prod_vals.mean():.3f}")
            stat_cols[1].metric("Moy. réf.", f"{ref_vals.mean():.3f}")
            stat_cols[2].metric("Méd. prod", f"{prod_vals.median():.3f}")
            stat_cols[3].metric("Méd. réf.", f"{ref_vals.median():.3f}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MISSING VALUES
# ════════════════════════════════════════════════════════════════════════════════
with tab_missing:
    st.subheader("Taux de valeurs manquantes (features nullables)")

    nullable_in_prod = [c for c in NULLABLE_FEATURES if c in successful_f.columns]
    if not nullable_in_prod:
        st.info("Aucune feature nullable trouvée dans les logs de production.")
    else:
        missing_rates = (
            successful_f[nullable_in_prod]
            .isna()
            .mean()
            .rename("missing_rate")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        missing_rates["alerte"] = missing_rates["missing_rate"] > 0.80

        st.bar_chart(missing_rates.set_index("feature")["missing_rate"])

        flagged = missing_rates[missing_rates["alerte"]]
        if not flagged.empty:
            st.error(
                f"🔴 {len(flagged)} feature(s) avec > 80 % de valeurs manquantes :"
            )
            st.dataframe(flagged, use_container_width=True)
        else:
            st.success(
                "✅ Toutes les features nullables sont dans les bornes attendues."
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — LOG EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("Explorateur de logs")

    col_filter, col_search = st.columns([2, 3])
    with col_filter:
        filter_success = st.radio(
            "Afficher",
            ["Tous", "Succès seulement", "Erreurs seulement"],
            horizontal=True,
        )
    with col_search:
        search_id = st.text_input(
            "Filtrer par request_id (partiel)", placeholder="uuid…"
        )

    if filter_success == "Succès seulement":
        view = pred_df_f[pred_df_f["success"]]
    elif filter_success == "Erreurs seulement":
        view = pred_df_f[~pred_df_f["success"]]
    else:
        view = pred_df_f

    if search_id:
        view = view[view["request_id"].str.contains(search_id, na=False)]

    display_cols = [
        "timestamp",
        "request_id",
        "success",
        "probability",
        "prediction_label",
        "latency_ms",
        "error",
    ] + [c for c in NUMERICAL_FEATURES if c in view.columns]
    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(
        view[display_cols].sort_values("timestamp", ascending=False).head(500),
        use_container_width=True,
    )
    st.caption(
        f"Affichage des 500 entrées les plus récentes (total filtré : {len(view):,})"
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — PERF
# ════════════════════════════════════════════════════════════════════════════════

with tab_perf:
    st.subheader("Performance Monitoring")

    if successful_f.empty:
        st.info("No performance data available.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Inference P50",
            f"{successful_f['inference_ms'].quantile(0.50):.1f} ms",
        )

        c2.metric(
            "Inference P95",
            f"{successful_f['inference_ms'].quantile(0.95):.1f} ms",
        )

        c3.metric(
            "CPU Avg",
            f"{successful_f['cpu_percent'].mean():.1f} %",
        )

        c4.metric(
            "RAM Avg",
            f"{successful_f['memory_mb'].mean():.0f} MB",
        )

        st.divider()

        st.markdown("### Inference time over time")

        inf_ts = (
            successful_f.set_index("timestamp")["inference_ms"].resample("1h").mean()
        )

        st.line_chart(inf_ts)

        st.markdown("### CPU usage over time")

        cpu_ts = (
            successful_f.set_index("timestamp")["cpu_percent"].resample("1h").mean()
        )

        st.line_chart(cpu_ts)

        st.markdown("### Memory usage over time")

        mem_ts = successful_f.set_index("timestamp")["memory_mb"].resample("1h").mean()

        st.line_chart(mem_ts)
