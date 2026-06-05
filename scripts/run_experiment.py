from credit_scoring.main import run_experiment

run_experiment(
    config="exp_lightgbm.yaml",
    load_raw_data=False,
    run_cv=True,
    run_feat_imp=True,
    run_shap=True,
    max_rows=None,
    debug=False,
)


# exp_lightgbm.yaml

# exp_xgboost.yaml

# exp_catboost.yaml

# exp_random_forest.yaml

# exp_logreg.yaml
