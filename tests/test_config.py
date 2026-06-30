from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    EDUCATION_INVERSE,
    EDUCATION_MAP,
    EDUCATION_OPTIONS,
    ENGINEERED_FEATURES,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    GENDER_INVERSE,
    GENDER_MAP,
    NUMERICAL_FEATURES,
    NULLABLE_FEATURES,
)


def test_feature_groups_cover_known_features():
    grouped = {feature for group in FEATURE_GROUPS.values() for feature in group}

    for feature in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert feature in grouped


def test_gender_mapping_is_bijective():
    assert set(GENDER_MAP) == {"M", "F"}
    assert GENDER_INVERSE == {v: k for k, v in GENDER_MAP.items()}


def test_education_mapping_is_consistent():
    assert EDUCATION_OPTIONS == list(EDUCATION_MAP)
    assert EDUCATION_INVERSE == {v: k for k, v in EDUCATION_MAP.items()}


def test_nullable_features_are_known_features():
    known = set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES) | set(ENGINEERED_FEATURES)
    assert set(NULLABLE_FEATURES).issubset(known)


def test_feature_labels_cover_main_inputs():
    for feature in CATEGORICAL_FEATURES + NUMERICAL_FEATURES[:5]:
        assert feature in FEATURE_LABELS
