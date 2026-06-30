import cProfile
import pstats

import numpy as np
import pandas as pd

from credit_scoring.serving.inference import get_model, get_reference_df


def run_perf_test():
    # On récupère le modèle et les données
    model, expected_features, _ = get_model()
    ref_df = get_reference_df()
    sample_features = ref_df.iloc[0].to_dict()

    # Préparation identique à votre méthode d'inférence
    X = pd.DataFrame([sample_features])[expected_features]
    X = X.replace({None: np.nan}).astype(float)

    # 50 itérations
    for _ in range(50):
        model.predict_proba(X)


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run_perf_test()
    pr.disable()

    # Affichage brut pour diagnostic
    stats = pstats.Stats(pr).sort_stats("tottime")
    stats.print_stats(20)  # Affiche les 20 plus lentes
