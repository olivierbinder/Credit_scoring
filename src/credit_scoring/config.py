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
FILE_DESC = DIR_DATA / "description" / "HomeCredit_columns_description.csv"

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
DF_PROC_PATH = DIR_DATA_PROCESSED / "df_proc.parquet"
PIPELINE_PATH = DIR_DATA_PROCESSED / "full_pipeline.pkl"

# MLFlow DB
MLFLOW_DB_PATH = DIR_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
PROD_MODEL_PATH = DIR_PROJ / "serving" / "model"
REF_DB_PATH = DIR_PROJ / "serving" / "db" / "reference.parquet"


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
    "📈 Credit Scores": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "👤 Applicant Profile": [
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "OWN_CAR_AGE",
    ],
    "💰 Loan Application": ["AMT_ANNUITY", "AMT_GOODS_PRICE", "PAYMENT_RATE"],
    "📅 Repayment History": [
        "INSTAL_DPD_MEAN",
        "INSTAL_AMT_PAYMENT_SUM",
        "POS_CNT_INSTALMENT_FUTURE_MEAN",
        "POS_SK_DPD_DEF_MEAN",
    ],
    "🏦 Credit History": [
        "PREV_CNT_PAYMENT_MEAN",
        "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
        "ACTIVE_DAYS_CREDIT_MAX",
    ],
    "💳 Credit Card Activity": [
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
    "EXT_SOURCE_1": "External Score 1",
    "EXT_SOURCE_2": "External Score 2",
    "EXT_SOURCE_3": "External Score 3",
    "CODE_GENDER": "Gender",
    "NAME_EDUCATION_TYPE": "Education",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "OWN_CAR_AGE": "Car Age",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_GOODS_PRICE": "Goods Price",
    "PAYMENT_RATE": "Payment Rate",
    "INSTAL_DPD_MEAN": "Avg Days Past Due",
    "INSTAL_AMT_PAYMENT_SUM": "Installment Payments",
    "POS_CNT_INSTALMENT_FUTURE_MEAN": "Future Installments",
    "POS_SK_DPD_DEF_MEAN": "POS Delinquency",
    "PREV_CNT_PAYMENT_MEAN": "Previous Payment Count",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN": "Previous Due Date",
    "ACTIVE_DAYS_CREDIT_MAX": "Active Credit Age",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": "ATM Withdrawals",
    "CC_CNT_DRAWINGS_CURRENT_VAR": "Card Usage Variability",
}
