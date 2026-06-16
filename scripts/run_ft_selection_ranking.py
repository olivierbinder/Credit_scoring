from credit_scoring.pipelines.feature_selection import (
    build_robust_feature_ranking,
)

ranking = build_robust_feature_ranking()

ranking.to_csv(
    "data/feature_selection/feature_ranking.csv",
    index=False,
)

print(ranking.head(50))
