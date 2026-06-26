# scripts/build_reference.py
import pandas as pd

from credit_scoring.config import DIR_DATA_PROCESSED, FILE_DATA_PROCESSED

FEATURES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "PAYMENT_RATE",
    "INSTAL_DPD_MEAN",
    "DAYS_EMPLOYED",
    "INSTAL_AMT_PAYMENT_SUM",
    "AMT_ANNUITY",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
    "CODE_GENDER",
    "OWN_CAR_AGE",
    "NAME_EDUCATION_TYPE",
    "DAYS_BIRTH",
    "PREV_CNT_PAYMENT_MEAN",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "ACTIVE_DAYS_CREDIT_MAX",
    "AMT_GOODS_PRICE",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
    "POS_SK_DPD_DEF_MEAN",
    "CC_CNT_DRAWINGS_CURRENT_VAR",
]


df = pd.read_parquet(FILE_DATA_PROCESSED)

ref = df[df["TARGET"].notnull()][["SK_ID_CURR"] + FEATURES]
ref.to_parquet(DIR_DATA_PROCESSED / "reference.parquet", index=False)

new = df[df["TARGET"].isnull()][["SK_ID_CURR"] + FEATURES]
new.to_parquet(DIR_DATA_PROCESSED / "new.parquet", index=False)
