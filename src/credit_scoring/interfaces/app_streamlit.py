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


def decode_for_display(raw: dict) -> dict:
    decoded = raw.copy()

    decoded["CODE_GENDER"] = GENDER_INVERSE.get(
        raw["CODE_GENDER"],
        "M",
    )

    decoded["NAME_EDUCATION_TYPE"] = EDUCATION_INVERSE.get(
        raw["NAME_EDUCATION_TYPE"],
        "Higher education",
    )

    return decoded


# PAGE CONFIG
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    layout="wide",
)

st.title("📊 Credit Scoring Predictor")
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data  # Très important pour ne pas recharger le fichier à chaque interaction
def load_reference_data():
    return pd.read_parquet(DIR_DATA_PROCESSED / "reference.parquet")


REFERENCE_DF = load_reference_data()


# PLOTTING HELPERS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def make_kde_plot(series, value):
    series = series.dropna()

    if len(series) < 10:
        return None

    if series.nunique() < 2:
        return None

    try:
        kde = gaussian_kde(series)
    except Exception:
        return None

    xs = np.linspace(
        series.min(),
        series.max(),
        200,
    )

    ys = kde(xs)

    fig = go.Figure()

    fig.add_scatter(
        x=xs,
        y=ys,
        mode="lines",
        fill="tozeroy",
    )

    if value is not None:
        fig.add_vline(
            x=float(value),
            line_color="red",
            line_width=3,
        )

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

    fig.update_xaxes(
        tickangle=0,
    )

    return fig


def get_reference_series(feature_name):
    series = REFERENCE_DF[feature_name]

    if feature_name == "CODE_GENDER":
        series = series.map(GENDER_INVERSE)

    elif feature_name == "NAME_EDUCATION_TYPE":
        series = series.map(EDUCATION_INVERSE)

    return series


# CLIENT LOADER
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
col_id, col_button = st.columns([3, 1])

with col_id:
    sk_id = st.number_input(
        "SK_ID_CURR",
        value=100002,
        min_value=0,
        step=1,
    )

with col_button:
    st.write("")
    st.write("")
    load_clicked = st.button(
        "Load client",
        use_container_width=True,
    )

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
        st.success("Client trouvé ✅")
    else:
        st.error("Client non trouvé ❌")

# DISPLAY MODE
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
edit_mode = st.toggle(
    "Edit mode",
    value=False,
)

# CLIENT DASHBOARD
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
if "features" in st.session_state:
    features = st.session_state["features"]

    st.divider()

    # READ ONLY MODE
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    if not edit_mode:
        st.subheader(f"Client {st.session_state['loaded_sk_id']}")

        for section_name, section_features in FEATURE_GROUPS.items():
            st.markdown(f"### {section_name}")

            for feature_name in section_features:
                if feature_name not in features:
                    continue

                value = features[feature_name]

                col_name, col_value, col_plot = st.columns([2, 2, 2])

                with col_name:
                    st.write(
                        FEATURE_LABELS.get(
                            feature_name,
                            feature_name,
                        )
                    )

                with col_value:
                    st.markdown(f"**{value if value is not None else 'NaN'}**")

                with col_plot:
                    series = get_reference_series(feature_name)

                    if feature_name in CATEGORICAL_FEATURES:
                        fig = make_bar_plot(
                            series,
                            value,
                        )
                    else:
                        fig = make_kde_plot(
                            series,
                            value,
                        )

                    if fig is not None:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "displayModeBar": False,
                            },
                        )

            st.divider()

    # EDIT MODE
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    else:
        st.subheader("Edit Features")

        edited = {}

        widget_prefix = str(st.session_state["loaded_sk_id"])

        for section_name, section_features in FEATURE_GROUPS.items():
            with st.expander(
                section_name,
                expanded=True,
            ):
                cols = st.columns(3)

                for idx, feature_name in enumerate(section_features):
                    if feature_name not in features:
                        continue

                    value = features[feature_name]

                    widget_key = f"{widget_prefix}_{feature_name}"

                    with cols[idx % 3]:
                        if feature_name == "CODE_GENDER":
                            edited[feature_name] = st.selectbox(
                                "Gender",
                                ["M", "F"],
                                index=["M", "F"].index(value),
                                key=widget_key,
                            )

                        elif feature_name == "NAME_EDUCATION_TYPE":
                            edited[feature_name] = st.selectbox(
                                "Education",
                                EDUCATION_OPTIONS,
                                index=EDUCATION_OPTIONS.index(value),
                                key=widget_key,
                            )

                        else:
                            txt = st.text_input(
                                feature_name,
                                value="" if value is None else str(value),
                                placeholder="NaN",
                                key=widget_key,
                            )

                            edited[feature_name] = None if txt == "" else float(txt)

        st.divider()

        predict_clicked = st.button(
            "Predict",
            use_container_width=True,
            type="primary",
        )

        if predict_clicked:
            with st.spinner("Computing prediction..."):
                response = requests.post(
                    f"{API_BASE}/predict",
                    json=edited,
                )

            if response.status_code != 200:
                st.error(f"API Error ({response.status_code})")

                st.code(response.text)

                st.stop()

            st.session_state["prediction"] = response.json()

    # PERSISTENT PREDICTION DISPLAY
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    if "prediction" in st.session_state:
        result = st.session_state["prediction"]

        st.divider()

        score_col, decision_col = st.columns(2)

        with score_col:
            st.metric(
                label="Default Probability",
                value=f"{result['probability']:.2%}",
            )

        with decision_col:
            st.metric(
                label="Prediction",
                value=result["prediction"],
            )
