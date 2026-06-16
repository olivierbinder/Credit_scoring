# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import functools
import time

import numpy as np
import pandas as pd

from credit_scoring.config import FILE_DESC
from credit_scoring.logger import logger


# TIMER
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def timer(func):
    """Decorator to time function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[{func.__name__}] starts.")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"[{func.__name__}] done in {end - start:.1f}s.")
        return result

    return wrapper


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


# PANDAS STYLING
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def display_df(
    df: pd.DataFrame,
    gradient_cols: list | None = None,
    mapping: dict | None = None,
    height: str = "400px",
    freeze_first_col: bool = True,
    max_col_width: str | None = None,
):
    """Style un DataFrame avec gradients, mappings, alignements et formatage numérique."""
    # Travail sur une copie pour éviter de modifier le DF original
    styler = df.style

    num_cols = df.select_dtypes(include=[np.number]).columns
    other_cols = df.columns.difference(num_cols)

    # 1. Appliquer les gradients en premier (sur les données brutes)
    if gradient_cols:
        valid_grads = [
            c for c in gradient_cols if c in df.columns and not df[c].dropna().empty
        ]
        if valid_grads:
            styler = styler.background_gradient(
                subset=valid_grads, cmap="YlOrRd", axis=0
            )
            # Rendre les 0 ou NaN transparents par-dessus le gradient
            styler = styler.map(
                lambda x: (
                    "background-color: transparent" if (pd.isna(x) or x == 0) else ""
                ),
                subset=valid_grads,
            )

    # 2. Appliquer les mappings de couleurs spécifiques
    if mapping:
        for col, val_map in mapping.items():
            if col in df.columns:
                styler = styler.map(
                    lambda x: (
                        f"background-color: {val_map.get(x)}; color: black"
                        if val_map.get(x)
                        else ""
                    ),
                    subset=[col],
                )

    # 3. Formater les nombres et les chaînes
    def _smart_num_format(x):
        if pd.isna(x) or x == 0:
            return ""
        return (
            f"{x:,.2f}".replace(",", " ")
            if isinstance(x, (float, np.floating))
            else f"{x:,}".replace(",", " ")
        )

    styler = styler.format(_smart_num_format, subset=num_cols)
    styler = styler.format(na_rep="", subset=other_cols)

    # 4. Définir les alignements
    styler = styler.set_properties(subset=other_cols, **{"text-align": "left"})  # ty:ignore[unresolved-attribute]
    styler = styler.set_properties(subset=num_cols, **{"text-align": "right"})

    # 5. Construction des styles CSS
    cell_props = [
        ("border", "1px solid #444"),
        ("padding", "6px"),
        ("font-size", "11px"),
        ("white-space", "nowrap"),
    ]
    if max_col_width:
        cell_props.extend(
            [
                ("max-width", max_col_width),
                ("overflow", "hidden"),
                ("text-overflow", "ellipsis"),
            ]
        )

    table_styles = [
        {
            "selector": "",
            "props": [
                ("max-height", height),
                ("display", "block"),
                ("overflow", "auto"),
                ("border", "1px solid #ccc"),
            ],
        },
        {
            "selector": "thead th",
            "props": [
                ("position", "sticky"),
                ("top", "0"),
                ("z-index", "2"),
                ("background-color", "#000"),
                ("color", "white"),
            ],
        },
        {"selector": "td, th", "props": cell_props},
    ]

    if freeze_first_col:
        # Figer la colonne index (TH)
        table_styles.extend(
            [
                {
                    "selector": "thead th:first-child",
                    "props": [
                        ("position", "sticky"),
                        ("left", "0"),
                        ("top", "0"),
                        ("z-index", "4"),
                        ("background-color", "#000"),
                        ("color", "white"),
                    ],
                },
                {
                    "selector": "tbody th:first-child",
                    "props": [
                        ("position", "sticky"),
                        ("left", "0"),
                        ("z-index", "3"),
                        ("background-color", "#000"),
                        ("color", "white"),
                    ],
                },
            ]
        )

    return styler.set_table_styles(table_styles)
