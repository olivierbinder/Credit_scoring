# src/credit_scoring/interfaces/app_streamlit.py
# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import gaussian_kde

from credit_scoring.config import DIR_DATA_PROCESSED
from credit_scoring.serving.constants import (
    CATEGORICAL_FEATURES,
    EDUCATION_INVERSE,
    EDUCATION_OPTIONS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    GENDER_INVERSE,
)

# CONFIGURATION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
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


# PAGE CONFIG
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .block-container { max-width: 1400px; padding-top: 3rem; }
        div[data-testid="stMetric"] { background: rgba(128,128,128,0.15); border-radius: 8px; padding: 0.75rem 1rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        div[data-testid="column"]:nth-child(2) { padding-right: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_reference_data():
    return pd.read_parquet(DIR_DATA_PROCESSED / "reference.parquet")


REFERENCE_DF = load_reference_data()


# PLOTTING HELPERS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def make_kde_plot(series, value):
    series = series.dropna()
    if len(series) < 10 or series.nunique() < 2:
        return None
    try:
        kde = gaussian_kde(series)
    except Exception:
        return None

    xs = np.linspace(series.min(), series.max(), 200)
    ys = kde(xs)

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


def make_bar_plot(series, value):
    counts = series.dropna().value_counts().sort_index()
    colors = [
        "crimson" if str(idx) == str(value) else "lightgray" for idx in counts.index
    ]
    LABELS = {
        "M": "M",
        "F": "F",
        "Lower secondary": "Lower<br>secondary",
        "Secondary / secondary special": "Secondary<br>/ secondary special",
        "Incomplete higher": "Incomplete<br>higher",
        "Higher education": "Higher<br>education",
        "Academic degree": "Academic<br>degree",
    }
    fig = go.Figure()
    fig.add_bar(
        x=[LABELS.get(str(x), str(x)) for x in counts.index],
        y=counts.values,
        marker_color=colors,
    )
    fig.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        yaxis_visible=False,
    )
    fig.update_xaxes(tickangle=0)
    return fig


def get_reference_series(feature_name):
    series = REFERENCE_DF[feature_name]
    if feature_name == "CODE_GENDER":
        series = series.map(GENDER_INVERSE)
    elif feature_name == "NAME_EDUCATION_TYPE":
        series = series.map(EDUCATION_INVERSE)
    return series


@st.cache_data
def make_kde_plot_cached(feature_name, value):
    series = get_reference_series(feature_name)
    return make_kde_plot(series, value)


@st.cache_data
def make_bar_plot_cached(feature_name, value):
    series = get_reference_series(feature_name)
    return make_bar_plot(series, value)


# TOP BAR — title + controls + prediction result
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
title_col, id_col, load_col, predict_col, result_col = st.columns([2, 2, 1, 1, 3])

with title_col:
    st.markdown("## 📊 Credit Scoring")

with id_col:
    st.write("")
    sk_id = st.number_input(
        "SK_ID_CURR", value=100002, min_value=0, step=1, label_visibility="collapsed"
    )

with load_col:
    st.write("")
    load_clicked = st.button("Load client", use_container_width=True)

with predict_col:
    st.write("")
    predict_clicked = st.button("Predict", use_container_width=True, type="primary")

with result_col:
    if "prediction" in st.session_state:
        result = st.session_state["prediction"]
        prob = result["probability"]
        is_default = result["prediction"] == "Likely to default"
        color = "inverse" if is_default else "normal"
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Default probability", f"{prob:.2%}", delta=None)
        with r2:
            st.metric("Decision", result["prediction"], delta=None, delta_color=color)

st.divider()

# LOAD CLIENT
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
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
        st.session_state.pop("prediction", None)  # reset stale prediction
        st.toast(f"Client {int(sk_id)} chargé ✅")
    else:
        st.error("Client non trouvé ❌")


# PREDICT
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
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


# FEATURE DISPLAY WITH INLINE EDITING
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def render_feature_group(group_name, features, edited, widget_prefix):
    """Render one feature group: name | editable value | distribution plot."""
    st.markdown(f"#### {group_name}")

    section_features = FEATURE_GROUPS[group_name]

    for feature_name in section_features:
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
            if feature_name in CATEGORICAL_FEATURES:
                fig = make_bar_plot_cached(feature_name, live_val)
            else:
                fig = make_kde_plot_cached(feature_name, live_val)
            if fig is not None:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    st.divider()


if "features" in st.session_state:
    features = st.session_state["features"]
    widget_prefix = str(st.session_state["loaded_sk_id"])

    # Initialise edited_features from loaded features on first render
    if "edited_features" not in st.session_state:
        st.session_state["edited_features"] = features.copy()

    # Use edited_features as source of truth — survives st.rerun()
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

    # Persist edited values so Predict can read them
    st.session_state["edited_features"] = edited
