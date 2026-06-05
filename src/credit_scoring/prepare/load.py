# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from pathlib import Path

import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split

# Project
from credit_scoring.config import (
    FILE_APP_TRAIN,
    FILE_DESC,
)
from credit_scoring.utils import timer

# LOAD AND SPLIT CORE FUNCTIONS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


def load_data(file: Path, num_rows: int | None = None) -> pd.DataFrame:
    """Load a single CSV file and print its shape."""
    df = pd.read_csv(file, nrows=num_rows)
    print(f"--> {file.name} loaded (shape = {df.shape[0]:,d} | {df.shape[1]:,d})")
    return df


def split_data(df: pd.DataFrame, target_col: str = "TARGET"):
    """Split dataframe into stratified train and test sets."""
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )

    X_train = train_df.drop(columns=[target_col]).reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)

    X_test = test_df.drop(columns=[target_col]).reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)
    print(
        f"--> Data split into train (n={X_train.shape[0]:,d}) and test (n={X_test.shape[0]:,d})"
    )
    return X_train, y_train, X_test, y_test


# LOAD AND SPLIT RECIPES
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
@timer
def load_and_split_baseline():
    """Load only the main application dataset."""
    applications = load_data(FILE_APP_TRAIN)
    return split_data(applications)


# DATA QUALITY ANALYSIS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def generate_data_quality_analysis(df_input, file_path):
    total_rows = len(df_input)
    diagnostic_data = []

    # Core Feature Metric Extraction Loop
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    for col in df_input.columns:
        series = df_input[col]
        dtype_str = str(series.dtype)

        # Base calculations
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / total_rows) * 100

        # Reset indicators to prevent cross-column contamination
        cat_info = None
        outlier_pct = np.nan
        zero_pct = np.nan
        top_values = None

        # Categorical or Object features
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == "object":
            cat_info = int(series.nunique())
            top_values = list(series.dropna().unique()[:5])

        # Numerical features
        elif np.issubdtype(series.dtype, np.number):
            valid_data = series.dropna()
            num_unique = valid_data.nunique()

            # Discrete numerical features
            if num_unique <= 10:
                cat_info = int(num_unique)
                top_values = [int(x) for x in valid_data.unique()[:5]]

            # Continuous numerical features
            else:
                zero_count = (valid_data == 0).sum()
                zero_pct = (zero_count / total_rows) * 100

                if len(valid_data) > 0:
                    q1 = valid_data.quantile(0.25)
                    q3 = valid_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outlier_count = (
                        (valid_data < lower_bound) | (valid_data > upper_bound)
                    ).sum()
                    outlier_pct = (outlier_count / total_rows) * 100
                else:
                    outlier_pct = 0.0

        # Build raw record
        diagnostic_data.append(
            {
                "Feature": col,
                "Dtype": dtype_str,
                "Cat": cat_info,
                "% Missing": missing_pct,
                "% Outliers": outlier_pct,
                "% Zeros": zero_pct,
                "Cat Values": top_values,
            }
        )

    # DataFrame Assembly & Formatting
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    df_result = pd.DataFrame(diagnostic_data)
    df_result["Cat"] = pd.to_numeric(df_result["Cat"], errors="coerce").astype("Int64")

    # Metadata Enrichment (Descriptions)
    # ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
    features_desc = pd.read_csv(FILE_DESC)
    if (
        file_path.name == "application_train.csv"
        or file_path.name == "application_test.csv"
    ):
        features_desc = features_desc.loc[
            features_desc["Table"] == "application_{train|test}.csv", :
        ]
    else:
        features_desc = features_desc.loc[features_desc["Table"] == file_path.name, :]

    df_result = df_result.merge(
        features_desc[["Row", "Description"]],
        left_on="Feature",
        right_on="Row",
        how="left",
    ).drop(columns=["Row"])

    return df_result
