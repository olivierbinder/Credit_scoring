# IMPORTS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import functools
import importlib
import time

import numpy as np
import pandas as pd
import yaml

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


# YAML LOADERS
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def load_yaml_config(config_path: str) -> dict:
    """Load YAML experiment configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_callable_function(module_path: str, function_name: str):
    """Dynamically import a function from a module path."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except (ModuleNotFoundError, AttributeError) as e:
        logger.error(f"Failed to load {module_path}.{function_name}: {e}")
        raise e


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
