import numpy as np
import pandas as pd

from credit_scoring.features.preprocess import preprocess_apps, sanitize_feature_names

# ── sanitize_feature_names ────────────────────────────────────────────────────


def test_sanitize_replaces_spaces():
    result = sanitize_feature_names(["hello world", "foo bar"])
    assert result == ["hello_world", "foo_bar"]


def test_sanitize_replaces_special_chars():
    result = sanitize_feature_names(["NAME_CONTRACT_STATUS_Cash loans"])
    assert "Cash_loans" in result[0]


def test_sanitize_collapses_multiple_underscores():
    result = sanitize_feature_names(["a  b"])  # double espace
    assert "__" not in result[0]


def test_sanitize_strips_leading_trailing_underscores():
    result = sanitize_feature_names([" leading", "trailing "])
    for name in result:
        assert not name.startswith("_")
        assert not name.endswith("_")


# ── preprocess_apps ───────────────────────────────────────────────────────────


def _make_apps(n=10) -> pd.DataFrame:
    """DataFrame minimal qui satisfait preprocess_apps."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CODE_GENDER": ["M", "F"] * (n // 2),
            "NAME_EDUCATION_TYPE": ["Higher education"] * n,
            "NAME_CONTRACT_TYPE": ["Cash loans"] * n,
            "DAYS_EMPLOYED": [365243] * (n // 2)
            + list(rng.integers(-3000, -100, n // 2)),
            "DAYS_BIRTH": list(rng.integers(-20000, -10000, n)),
            "AMT_INCOME_TOTAL": list(rng.uniform(50_000, 200_000, n)),
            "AMT_CREDIT": list(rng.uniform(100_000, 500_000, n)),
            "AMT_ANNUITY": list(rng.uniform(5_000, 30_000, n)),
            "CNT_FAM_MEMBERS": list(rng.integers(1, 5, n).astype(float)),
        }
    )


def test_preprocess_apps_removes_xna_gender():
    df = _make_apps(10)
    df.loc[0, "CODE_GENDER"] = "XNA"
    result = preprocess_apps(df)
    # la ligne XNA doit être supprimée
    assert len(result) == 9


def test_preprocess_apps_replaces_days_employed_sentinel():
    df = _make_apps(10)
    result = preprocess_apps(df)
    assert result["DAYS_EMPLOYED"].isna().any()


def test_preprocess_apps_creates_ratio_columns():
    df = _make_apps(10)
    result = preprocess_apps(df)
    expected = [
        "DAYS_EMPLOYED_PERC",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
    ]
    for col in expected:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_preprocess_apps_no_inf():
    df = _make_apps(10)
    result = preprocess_apps(df)
    num = result.select_dtypes(include="number")
    assert not np.isinf(num.values).any()
