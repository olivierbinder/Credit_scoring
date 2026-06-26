# src/credit_scoring/serving/constants.py

# %%  REFERENCE DATA                                                                   .

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


# %%  CATEGORICAL ENCODING                                                             .


CATEGORICAL_FEATURES = {
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
}
# Education Options for SelectBoxes
EDUCATION_OPTIONS = [
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]
