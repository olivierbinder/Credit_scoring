# src/credit_scoring/interfaces/app_streamlit.py
"""
Point d'entrée de l'application Streamlit multi-pages.

Lancement :
    streamlit run src/credit_scoring/interfaces/app_streamlit.py

Pages :
    📊 Scoring     — prédiction interactive par client
    📡 Monitoring  — drift, anomalies opérationnelles, logs
"""

import streamlit as st

st.set_page_config(
    page_title="Credit Scoring",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 1400px; padding-top: 3rem; }
        div[data-testid="stMetric"] {
            background: rgba(128,128,128,0.15);
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        div[data-testid="column"]:nth-child(2) { padding-right: 2rem; }
        div[data-testid="stNumberInput"] input { font-size: 0.9rem; }
        div[data-testid="stNumberInput"] { margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pages ──────────────────────────────────────────────────────────────────────
scoring_page = st.Page("pages/scoring.py", title="Scoring", icon="📊", default=True)
monitoring_page = st.Page("pages/monitoring.py", title="Monitoring", icon="📡")

pg = st.navigation([scoring_page, monitoring_page])
pg.run()
