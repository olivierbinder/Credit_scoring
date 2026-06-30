# %%  IMPORTS                                                                          .
from pathlib import Path

# %%  SETTINGS                                                                         .
RANDOM_STATE: int = 42
DTYPE_COLORS: dict[str, str] = {
    "float64": "#72efdd",
    "int64": "#64c6df",
    "object": "#ffb5a7",
    "bool": "#f5fbb9",
}
# %%  # PATHS                                                                          .
# Project root
DIR_ROOT = Path(__file__).resolve().parents[2]

# Directories
DIR_DATA = DIR_ROOT / "data"
DIR_DATA_RAW = DIR_DATA / "raw"
DIR_DATA_PROCESSED = DIR_DATA / "processed"
DIR_CONFIG = DIR_ROOT / "config"
DIR_PROJ = DIR_ROOT / "src" / "credit_scoring"

# Desciption file
DESC_RAW_FILES = DIR_DATA / "description" / "HomeCredit_columns_description.csv"

# Raw files
FILE_APP_TEST = DIR_DATA_RAW / "application_test.csv"
FILE_APP_TRAIN = DIR_DATA_RAW / "application_train.csv"
FILE_BUREAU = DIR_DATA_RAW / "bureau.csv"
FILE_BUREAU_BALANCE = DIR_DATA_RAW / "bureau_balance.csv"
FILE_PREVIOUS_APP = DIR_DATA_RAW / "previous_application.csv"
FILE_PREVIOUS_BALANCE_CARD = DIR_DATA_RAW / "credit_card_balance.csv"
FILE_PREVIOUS_BALANCE_CASH = DIR_DATA_RAW / "POS_CASH_balance.csv"
FILE_PREVIOUS_PAYMENTS = DIR_DATA_RAW / "installments_payments.csv"

# Processed files
X_TRAIN_PROC_PATH = DIR_DATA_PROCESSED / "X_train_proc.csv"
X_TEST_PROC_PATH = DIR_DATA_PROCESSED / "X_test_proc.csv"
Y_TRAIN_PROC_PATH = DIR_DATA_PROCESSED / "y_train.csv"
Y_TEST_PROC_PATH = DIR_DATA_PROCESSED / "y_test.csv"
FILE_DATA_PROCESSED = DIR_DATA_PROCESSED / "df_proc.parquet"
PIPELINE_PATH = DIR_DATA_PROCESSED / "full_pipeline.pkl"

# MLFlow DB
ML_FLOW_DB = DIR_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{ML_FLOW_DB}"
PROD_MODEL = DIR_PROJ / "serving" / "model"
PROD_REFERENCE = DIR_PROJ / "serving" / "db" / "reference.parquet"
PROD_TEST = DIR_PROJ / "serving" / "db" / "test.parquet"


# Default paths — overridable via st.session_state set by the main app
FILE_PRED = DIR_ROOT / "logs/predictions.jsonl"
FILE_API = DIR_ROOT / "logs/api_calls.jsonl"
FILE_REFERENCE = DIR_ROOT / "data/processed/reference.parquet"
FILE_DRIFT_REPORT = DIR_ROOT / "reports/drift_report.html"
FILE_QUALITY_REPORT = DIR_ROOT / "reports/quality_report.html"
# %%  # FEATURES                                                                       .
# Final model features
NUMERICAL_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "PAYMENT_RATE",
    "OWN_CAR_AGE",
    "INSTAL_DPD_MEAN",
    "INSTAL_AMT_PAYMENT_SUM",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
    "POS_SK_DPD_DEF_MEAN",
    "PREV_CNT_PAYMENT_MEAN",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
    "ACTIVE_DAYS_CREDIT_MAX",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "CC_CNT_DRAWINGS_CURRENT_VAR",
]

CATEGORICAL_FEATURES = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

NULLABLE_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_3",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE",
    "ACTIVE_DAYS_CREDIT_MAX",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "CC_CNT_DRAWINGS_CURRENT_VAR",
]
# Cat Values
EDUCATION_OPTIONS = [
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]

# Engineered Features
ENGINEERED_FEATURES = [
    "INSTAL_DPD_MEAN",
    "INSTAL_AMT_PAYMENT_SUM",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
    "POS_SK_DPD_DEF_MEAN",
    "PREV_CNT_PAYMENT_MEAN",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
    "ACTIVE_DAYS_CREDIT_MAX",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "CC_CNT_DRAWINGS_CURRENT_VAR",
]

# Feature Groups for Dashboard Layout
FEATURE_GROUPS = {
    "📈 Scores externes de solvabilité": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ],
    "👤 Profil du demandeur": [
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "OWN_CAR_AGE",
    ],
    "💰 Caractéristiques du prêt": ["AMT_ANNUITY", "AMT_GOODS_PRICE", "PAYMENT_RATE"],
    "📅 Historique de remboursement": [
        "INSTAL_DPD_MEAN",
        "INSTAL_AMT_PAYMENT_SUM",
        "POS_CNT_INSTALMENT_FUTURE_MEAN",
        "POS_SK_DPD_DEF_MEAN",
    ],
    "🏦 Historique de crédit": [
        "PREV_CNT_PAYMENT_MEAN",
        "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
        "ACTIVE_DAYS_CREDIT_MAX",
    ],
    "💳 Utilisation carte de crédit": [
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
        "CC_CNT_DRAWINGS_CURRENT_VAR",
    ],
}


# Mappings
GENDER_MAP = {
    "M": 1.0,
    "F": 0.0,
}

EDUCATION_MAP = {
    "Lower secondary": 0.0,
    "Secondary / secondary special": 1.0,
    "Incomplete higher": 2.0,
    "Higher education": 3.0,
    "Academic degree": 4.0,
}

GENDER_INVERSE = {v: k for k, v in GENDER_MAP.items()}
EDUCATION_INVERSE = {v: k for k, v in EDUCATION_MAP.items()}

FEATURE_LABELS = {
    # Scores externes
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    # Profil du demandeur
    "CODE_GENDER": "Genre du demandeur",
    "NAME_EDUCATION_TYPE": "Niveau d’études",
    "DAYS_BIRTH": "Âge du demandeur (jours)",
    "DAYS_EMPLOYED": "Ancienneté professionnelle (jours)",
    "OWN_CAR_AGE": "Ancienneté du véhicule (années)",
    # Caractéristiques du prêt
    "AMT_ANNUITY": "Mensualité prévue (€)",
    "AMT_GOODS_PRICE": "Prix du bien financé (€)",
    "PAYMENT_RATE": "Ratio mensualité / montant du crédit",
    # Historique de remboursement
    "INSTAL_DPD_MEAN": "Retard moyen de paiement (jours)",
    "INSTAL_AMT_PAYMENT_SUM": "Montant total remboursé (€)",
    "POS_CNT_INSTALMENT_FUTURE_MEAN": "Échéances restantes moyennes (mois)",
    "POS_SK_DPD_DEF_MEAN": "Retard moyen avec tolérance (jours)",
    # Historique de crédit
    "PREV_CNT_PAYMENT_MEAN": "Nombre moyen d’échéances des anciens crédits (mois)",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN": "Fin prévue moyenne des anciens crédits (jours)",
    "ACTIVE_DAYS_CREDIT_MAX": "Ancienneté du dernier crédit actif (jours)",
    # Carte de crédit
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": "Nombre moyen de retraits au distributeur par mois",
    "CC_CNT_DRAWINGS_CURRENT_VAR": "Variabilité mensuelle du nombre d’opérations carte",
}
