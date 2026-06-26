from pathlib import Path

from credit_scoring.features.feature_selection import (
    evaluate_feature_subsets,
)

results = evaluate_feature_subsets(
    ranking_path=Path("data/feature_selection/feature_ranking.csv"),
    min_features=3,
    max_features=25,
)

results.to_csv(
    "data/feature_selection/feature_subset_results.csv",
    index=False,
)

print(results[["n_features", "roc_auc", "business_cost"]])
