# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

import gc
import re

import numpy as np
import pandas as pd

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
)

from credit_scoring.config import (
    DIR_DATA,
    FILE_APP_TEST,
    FILE_APP_TRAIN,
    FILE_BUREAU,
    FILE_BUREAU_BALANCE,
    FILE_PREVIOUS_APP,
    FILE_PREVIOUS_BALANCE_CARD,
    FILE_PREVIOUS_BALANCE_CASH,
    FILE_PREVIOUS_PAYMENTS,
)
from credit_scoring.data.load import load_data
from credit_scoring.logger import logger
from credit_scoring.utils import timer


# PREPROCESSING APPS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@timer
def preprocess_apps(apps):
    # Cleaning
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    df = apps.copy()
    df = df[df["CODE_GENDER"] != "XNA"]
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Categorical features
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    col_cat_edu_order = [
        "Lower secondary",
        "Secondary / secondary special",
        "Incomplete higher",
        "Higher education",
        "Academic degree",
    ]

    all_cat_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    edu_col = ["NAME_EDUCATION_TYPE"]
    other_cat_cols = [c for c in all_cat_cols if c not in edu_col]

    # Education pipeline
    cat_edu_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["Unknown"] + col_cat_edu_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    # Default pipeline
    cat_default_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Column Transformer
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_edu", cat_edu_pipe, edu_col),
            ("cat_default", cat_default_pipe, other_cat_cols),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",  # Keeps numeric columns as they are
    )

    # Apply transformation
    transformed_data = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()
    df_proc = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)

    # Ratios
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    df_proc["DAYS_EMPLOYED_PERC"] = df_proc["DAYS_EMPLOYED"] / df_proc["DAYS_BIRTH"]
    df_proc["INCOME_CREDIT_PERC"] = df_proc["AMT_INCOME_TOTAL"] / df_proc["AMT_CREDIT"]
    df_proc["INCOME_PER_PERSON"] = (
        df_proc["AMT_INCOME_TOTAL"] / df_proc["CNT_FAM_MEMBERS"]
    )
    df_proc["ANNUITY_INCOME_PERC"] = (
        df_proc["AMT_ANNUITY"] / df_proc["AMT_INCOME_TOTAL"]
    )
    df_proc["PAYMENT_RATE"] = df_proc["AMT_ANNUITY"] / df_proc["AMT_CREDIT"]

    return df_proc


# PREPROCESS BUREAU
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@timer
def preprocess_bureau(bureau, bb):
    # Encoding setup
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    cat_cols_bb = bb.select_dtypes(include=["object"]).columns.tolist()
    cat_cols_bur = bureau.select_dtypes(include=["object"]).columns.tolist()

    ohe_bb = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cat_cols_bb,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ohe_bur = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cat_cols_bur,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    bb_enc = pd.DataFrame(
        ohe_bb.fit_transform(bb), columns=ohe_bb.get_feature_names_out(), index=bb.index
    )
    bur_enc = pd.DataFrame(
        ohe_bur.fit_transform(bureau),
        columns=ohe_bur.get_feature_names_out(),
        index=bureau.index,
    )

    # Bureau Balance Aggregation
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    bb_agg = bb_enc.groupby("SK_ID_BUREAU").agg(
        {
            **{"MONTHS_BALANCE": ["min", "max", "size"]},
            **{
                col: ["mean"]
                for col in bb_enc.columns
                if col not in ["SK_ID_BUREAU", "MONTHS_BALANCE"]
            },
        }
    )
    bb_agg.columns = [f"{e[0]}_{e[1].upper()}" for e in bb_agg.columns]

    bureau = bur_enc.merge(bb_agg, how="left", on="SK_ID_BUREAU").drop(
        columns=["SK_ID_BUREAU"]
    )
    del bb, bb_enc, bur_enc, bb_agg
    gc.collect()

    # Bureau Aggregations
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    num_aggs = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }

    cat_aggs = {
        col: ["mean"]
        for col in bureau.columns
        if col not in num_aggs and col != "SK_ID_CURR"
    }

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggs, **cat_aggs})
    bureau_agg.columns = [f"BURO_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns]

    # Active/Closed Splits
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    for status in ["Active", "Closed"]:
        subset = bureau[bureau[f"CREDIT_ACTIVE_{status}"] == 1]
        sub_agg = subset.groupby("SK_ID_CURR").agg(num_aggs)
        sub_agg.columns = [
            f"{status.upper()}_{e[0]}_{e[1].upper()}" for e in sub_agg.columns
        ]
        bureau_agg = bureau_agg.join(sub_agg, how="left", on="SK_ID_CURR")
        del subset, sub_agg
        gc.collect()

    return bureau_agg


# PREPROCESSING OF PREVIOUS APPLICATIONS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@timer
def preprocess_previous_applications(prev):
    prev = prev.drop(columns=["SK_ID_PREV"])
    # Cleaning & Feature Engineering
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    days_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for col in days_cols:
        prev[col] = prev[col].replace(365243, np.nan)
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    # Encoding
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    cat_cols = prev.select_dtypes(include=["object"]).columns.tolist()
    ohe = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    prev_enc = pd.DataFrame(
        ohe.fit_transform(prev), columns=ohe.get_feature_names_out(), index=prev.index
    )

    # Aggregations
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    num_aggs = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cat_aggs = {
        col: ["mean"]
        for col in prev_enc.columns
        if col not in num_aggs and col != "SK_ID_CURR"
    }

    prev_agg = prev_enc.groupby("SK_ID_CURR").agg({**num_aggs, **cat_aggs})
    prev_agg.columns = [f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns]

    for status in ["Approved", "Refused"]:
        subset = prev_enc[prev_enc[f"NAME_CONTRACT_STATUS_{status}"] == 1]
        sub_agg = subset.groupby("SK_ID_CURR").agg(num_aggs)
        sub_agg.columns = [
            f"{status.upper()}_{e[0]}_{e[1].upper()}" for e in sub_agg.columns
        ]
        prev_agg = prev_agg.join(sub_agg, how="left", on="SK_ID_CURR")

    return prev_agg


@timer
def preprocess_pos_cash(pos):
    pos = pos.drop(columns=["SK_ID_PREV"])
    ohe = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                pos.select_dtypes(include=["object"]).columns.tolist(),
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pos_enc = pd.DataFrame(
        ohe.fit_transform(pos), columns=ohe.get_feature_names_out(), index=pos.index
    )

    aggs = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    aggs.update(
        {
            col: ["mean"]
            for col in pos_enc.columns
            if col not in aggs and col != "SK_ID_CURR"
        }
    )

    pos_agg = pos_enc.groupby("SK_ID_CURR").agg(aggs)
    pos_agg.columns = [f"POS_{e[0]}_{e[1].upper()}" for e in pos_agg.columns]
    pos_agg["POS_COUNT"] = pos_enc.groupby("SK_ID_CURR").size()
    return pos_agg


def preprocess_installments(ins):
    ins = ins.drop(columns=["SK_ID_PREV"])
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    ohe = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                ins.select_dtypes(include=["object"]).columns.tolist(),
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ins_enc = pd.DataFrame(
        ohe.fit_transform(ins), columns=ohe.get_feature_names_out(), index=ins.index
    )

    aggs = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    aggs.update(
        {
            col: ["mean"]
            for col in ins_enc.columns
            if col not in aggs and col != "SK_ID_CURR"
        }
    )

    ins_agg = ins_enc.groupby("SK_ID_CURR").agg(aggs)
    ins_agg.columns = [f"INSTAL_{e[0]}_{e[1].upper()}" for e in ins_agg.columns]
    ins_agg["INSTAL_COUNT"] = ins_enc.groupby("SK_ID_CURR").size()
    return ins_agg


@timer
def preprocess_credit_card(cc):
    cc = cc.drop(columns=["SK_ID_PREV"])
    ohe = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cc.select_dtypes(include=["object"]).columns.tolist(),
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    cc_enc = pd.DataFrame(
        ohe.fit_transform(cc), columns=ohe.get_feature_names_out(), index=cc.index
    )

    cc_agg = cc_enc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = [f"CC_{e[0]}_{e[1].upper()}" for e in cc_agg.columns]
    cc_agg["CC_COUNT"] = cc_enc.groupby("SK_ID_CURR").size()
    return cc_agg


def sanitize_feature_names(names) -> list[str]:
    """Normalize feature names for tree/boosting models (spaces, punctuation -> _)."""
    clean = [re.sub(r"[ \s,:/\\\-]+", "_", str(n)) for n in names]
    clean = [re.sub(r"_{2,}", "_", c) for c in clean]
    return [c.strip("_") for c in clean]


# MAIN PREPROCESSING FUNCTION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def load_and_preprocess_all(num_rows):
    """Load, preprocess, and merge all datasets into a single dataframe."""
    apps_train = load_data(FILE_APP_TRAIN, num_rows)
    apps_test = load_data(FILE_APP_TEST, num_rows)
    df = pd.concat(objs=[apps_train, apps_test], axis=0, ignore_index=True)
    del apps_train, apps_test
    gc.collect()
    df = preprocess_apps(df)

    # Load and preprocess bureau data
    bureau = load_data(FILE_BUREAU, num_rows)
    bb = load_data(FILE_BUREAU_BALANCE, num_rows)
    bureau_agg = preprocess_bureau(bureau, bb)
    df = df.join(bureau_agg, how="left", on="SK_ID_CURR")
    del bureau, bb, bureau_agg
    gc.collect()

    # Load and preprocess previous applications data
    prev = load_data(FILE_PREVIOUS_APP, num_rows)
    prev_agg = preprocess_previous_applications(prev)
    df = df.join(prev_agg, how="left", on="SK_ID_CURR")
    del prev, prev_agg
    gc.collect()

    pb = load_data(FILE_PREVIOUS_BALANCE_CASH, num_rows)
    pos_agg = preprocess_pos_cash(pb)
    df = df.join(pos_agg, how="left", on="SK_ID_CURR")
    del pb, pos_agg
    gc.collect()

    pp = load_data(FILE_PREVIOUS_PAYMENTS, num_rows)
    ins_agg = preprocess_installments(pp)
    df = df.join(ins_agg, how="left", on="SK_ID_CURR")
    del pp, ins_agg
    gc.collect()

    pc = load_data(FILE_PREVIOUS_BALANCE_CARD, num_rows)
    cc_agg = preprocess_credit_card(pc)
    df = df.join(cc_agg, how="left", on="SK_ID_CURR")
    del pc, cc_agg
    gc.collect()

    df = df.replace([np.inf, -np.inf], np.nan)
    df.columns = sanitize_feature_names(df.columns)
    logger.info(
        f"🆗 Full dataset loaded and preprocessed (shape = {df.shape[0]:,d} | {df.shape[1]:,d})"
    )

    return df


# FEATURE SELECTION
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def select_important_features(df, n=20):
    ft = pd.read_csv(DIR_DATA / "feature_selection" / "feature_ranking.csv")
    top_features = ft["feature"].head(n).tolist()
    return df[top_features]
