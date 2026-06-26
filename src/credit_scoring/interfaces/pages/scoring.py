# src/credit_scoring/interfaces/pages/scoring.py
"""
Page « Scoring » — prédiction interactive par client.
Contenu original de app_streamlit.py, sans set_page_config ni CSS
(gérés dans le fichier d'entrée app_streamlit.py).
"""

# %% IMPORTS                                                                           .

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import gaussian_kde

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    EDUCATION_INVERSE,
    EDUCATION_OPTIONS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    GENDER_INVERSE,
)

# %% CONFIG                                                                            .

API_BASE = "http://127.0.0.1:8000"

_all_groups = list(FEATURE_GROUPS.keys())
LEFT_GROUPS = _all_groups[:3]
RIGHT_GROUPS = _all_groups[3:]


def decode_for_display(raw: dict) -> dict:
    decoded = raw.copy()
    decoded["CODE_GENDER"] = GENDER_INVERSE.get(raw["CODE_GENDER"], "M")
    decoded["NAME_EDUCATION_TYPE"] = EDUCATION_INVERSE.get(
        raw["NAME_EDUCATION_TYPE"], "Higher education"
    )
    return decoded


# %% CACHE                                                                             .


@st.cache_data
def load_model_info():
    response = requests.get(f"{API_BASE}/model-info")
    response.raise_for_status()
    return response.json()


THRESHOLD = load_model_info()["threshold"]


@st.cache_data
def load_reference_data():
    response = requests.get(f"{API_BASE}/reference")
    response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


REFERENCE_DF = load_reference_data()


# %%  PLOTTING HELPERS                                                                 .


def get_reference_series(feature_name):
    series = REFERENCE_DF[feature_name]
    if feature_name == "CODE_GENDER":
        series = series.map(GENDER_INVERSE)
    elif feature_name == "NAME_EDUCATION_TYPE":
        series = series.map(EDUCATION_INVERSE)
    return series


@st.cache_data
def get_kde_data(feature_name):
    series = get_reference_series(feature_name).dropna()
    if len(series) < 10 or series.nunique() < 2:
        return None
    try:
        kde = gaussian_kde(series)
    except Exception:
        return None
    xs = np.linspace(series.min(), series.max(), 200)
    return xs.tolist(), kde(xs).tolist()


@st.cache_data
def get_bar_data(feature_name):
    series = get_reference_series(feature_name).dropna()
    counts = series.value_counts().sort_index()
    return counts.index.tolist(), counts.values.tolist()


def make_kde_plot_for_value(feature_name, value):
    data = get_kde_data(feature_name)
    if data is None:
        return None
    xs, ys = data
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", fill="tozeroy")
    if value is not None:
        fig.add_vline(x=float(value), line_color="red", line_width=3)
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis_visible=False,
        yaxis_visible=False,
    )
    return fig


def make_bar_plot_for_value(feature_name, value):
    data = get_bar_data(feature_name)
    if data is None:
        return None
    index, values = data
    LABELS = {
        "M": "M",
        "F": "F",
        "Lower secondary": "Low. Sec",
        "Secondary / secondary special": "Sec.",
        "Incomplete higher": "Inc.",
        "Higher education": "Higher",
        "Academic degree": "Academic",
    }
    colors = ["crimson" if str(i) == str(value) else "lightgray" for i in index]
    fig = go.Figure()
    fig.add_bar(
        x=[LABELS.get(str(x), str(x)) for x in index], y=values, marker_color=colors
    )
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=20),
        showlegend=False,
        yaxis_visible=False,
    )
    fig.update_xaxes(tickangle=0, tickfont_size=9)
    return fig


# %%          .          TOP BAR           .


title_col, sep, id_col, load_col, predict_col, result_col = st.columns(
    [2, 0.1, 1, 0.8, 0.8, 4]
)

with title_col:
    st.markdown("### 📊 Credit Scoring")

with id_col:
    sk_id = st.number_input("ID Client", value=100002, min_value=0, step=1)

with load_col:
    st.write("")
    load_clicked = st.button("Load", use_container_width=True)

with predict_col:
    st.write("")
    predict_clicked = st.button("Predict", use_container_width=True, type="primary")

with result_col:
    if "prediction" in st.session_state:
        result = st.session_state["prediction"]
        prob = result["probability"]
        is_default = prob >= THRESHOLD

        gauge_col, label_col = st.columns([1.5, 1])

        with gauge_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 22}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"size": 9}},
                        "bar": {"color": "crimson" if is_default else "steelblue"},
                        "steps": [
                            {
                                "range": [0, THRESHOLD * 100],
                                "color": "rgba(0,128,0,0.15)",
                            },
                            {
                                "range": [THRESHOLD * 100, 100],
                                "color": "rgba(255,0,0,0.15)",
                            },
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.85,
                            "value": THRESHOLD * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=0))
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        with label_col:
            st.write("")
            st.write("")
            icon = "⚠️" if is_default else "✅"
            color = "crimson" if is_default else "steelblue"
            verdict = "Likely to default" if is_default else "Not likely to default"
            st.markdown(
                f"""
                <div style="text-align:center; line-height:1.6">
                    <div style="font-size:2rem">{icon}</div>
                    <div style="font-size:0.85rem; font-weight:600; color:{color}">{verdict}</div>
                    <div style="font-size:0.75rem; color:gray; margin-top:4px">
                        Threshold: <b>{round(THRESHOLD * 100, 2)}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()


# %%          .          LOAD CLIENT          .

if load_clicked:
    try:
        response = requests.get(f"{API_BASE}/lookup/{int(sk_id)}", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de joindre l'API : {e}")
        st.stop()

    if response.status_code == 200:
        st.session_state["features"] = decode_for_display(response.json())
        st.session_state["loaded_sk_id"] = int(sk_id)
        st.session_state.pop("prediction", None)
        st.session_state.pop("edited_features", None)
        st.toast(f"Client {int(sk_id)} chargé ✅")
    else:
        st.error("Client non trouvé ❌")

# %%          .          PREDICT          .


if predict_clicked:
    if "edited_features" not in st.session_state:
        st.warning("Chargez un client d'abord.")
    else:
        with st.spinner("Computing prediction..."):
            response = requests.post(
                f"{API_BASE}/predict",
                json=st.session_state["edited_features"],
            )
        if response.status_code != 200:
            st.error(f"API Error ({response.status_code})")
            st.code(response.text)
            st.stop()
        st.session_state["prediction"] = response.json()
        st.rerun()

# %%          .          FEATURE DISPLAY WITH INLINE EDITING          .


def render_feature_group(group_name, features, edited, widget_prefix):
    st.markdown(f"#### {group_name}")
    for feature_name in FEATURE_GROUPS[group_name]:
        if feature_name not in features:
            continue
        value = features[feature_name]
        widget_key = f"{widget_prefix}_{feature_name}"

        col_name, col_widget, col_plot = st.columns([2, 2, 3])

        with col_name:
            st.markdown(
                f"<div style='padding-top:0.45rem;font-size:0.85rem;color:#555'>"
                f"{FEATURE_LABELS.get(feature_name, feature_name)}</div>",
                unsafe_allow_html=True,
            )
        with col_widget:
            if feature_name == "CODE_GENDER":
                new_val = st.selectbox(
                    "Gender",
                    ["M", "F"],
                    index=["M", "F"].index(value) if value in ["M", "F"] else 0,
                    key=widget_key,
                    label_visibility="collapsed",
                )
            elif feature_name == "NAME_EDUCATION_TYPE":
                new_val = st.selectbox(
                    "Education",
                    EDUCATION_OPTIONS,
                    index=EDUCATION_OPTIONS.index(value)
                    if value in EDUCATION_OPTIONS
                    else 0,
                    key=widget_key,
                    label_visibility="collapsed",
                )
            else:
                txt = st.text_input(
                    feature_name,
                    value="" if value is None else str(value),
                    placeholder="NaN",
                    key=widget_key,
                    label_visibility="collapsed",
                )
                new_val = None if txt == "" else float(txt)
            edited[feature_name] = new_val

        with col_plot:
            live_val = edited.get(feature_name, value)
            fig = (
                make_bar_plot_for_value(feature_name, live_val)
                if feature_name in CATEGORICAL_FEATURES
                else make_kde_plot_for_value(feature_name, live_val)
            )
            if fig is not None:
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

    st.divider()


if "features" in st.session_state:
    features = st.session_state["features"]
    widget_prefix = str(st.session_state["loaded_sk_id"])

    if "edited_features" not in st.session_state:
        st.session_state["edited_features"] = features.copy()

    current_features = st.session_state["edited_features"]
    edited = {}

    left_col, gap_col, right_col = st.columns([10, 1, 10])

    with left_col:
        for group_name in LEFT_GROUPS:
            if group_name in FEATURE_GROUPS:
                render_feature_group(
                    group_name, current_features, edited, widget_prefix
                )

    with right_col:
        for group_name in RIGHT_GROUPS:
            if group_name in FEATURE_GROUPS:
                render_feature_group(
                    group_name, current_features, edited, widget_prefix
                )

    st.session_state["edited_features"] = edited
