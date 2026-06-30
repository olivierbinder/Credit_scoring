# Conception du modèle

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
Le travail modèle vise un résultat <strong>simple à servir</strong> : 20 variables finales, un seuil métier, un LightGBM packagé.
</div>

## Préparation des données

[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

```mermaid
flowchart TD
    A[Tables Home Credit] --> B[Nettoyage application]
    A --> C[Agrégations bureau]
    A --> D[Agrégations historiques]
    B --> E[Dataset client]
    C --> E
    D --> E
    E --> F[Sélection de features]
    F --> G[Optimisation du modèle]
    G --> H[LightGBM production]
```

## Arbitrages importants

- Les tables brutes sont ramenées au niveau **client `SK_ID_CURR`**.
- Le pipeline crée des variables métier : ratios crédit/revenu, paiement, retards, historiques.
- La sélection réduit **plus de 600 variables** à **20 features** pour simplifier le serving.
- Le seuil est choisi avec un **business score** qui pénalise fortement les faux négatifs.

## Entraînement et suivi

[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

| Élément | Rôle |
| --- | --- |
| `config/training.yaml` | Reproduire les expériences |
| MLflow | Tracer paramètres, métriques et seuil |
| LightGBM | Modèle retenu pour production |
| ONNX | Version optimisée pour benchmark |

## Sortie modèle

```json
{
  "probability": 0.1842,
  "prediction": "Not likely to default"
}
```

??? info "Annexes"

    ## Données utilisées

    - Tables principales : `application_train` et `application_test`.
    - Crédits externes : `bureau` et `bureau_balance`.
    - Historiques précédents : `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`.

    ## Préprocessing détaillé

    - Nettoyage de valeurs métier comme `DAYS_EMPLOYED = 365243` et `CODE_GENDER = XNA`.
    - Encodage ordinal du niveau d'études et one-hot encoding des autres catégories.
    - Création de ratios métier : `PAYMENT_RATE`, revenu/crédit, revenu/personne.
    - Agrégations par client : moyennes, sommes, variances, séparation `Active` / `Closed`.
    - Fusion finale de toutes les sources dans un dataset tabulaire client.

    ## Entraînement

    - Plusieurs familles de modèles peuvent être comparées : LightGBM, XGBoost, CatBoost, Random Forest, régression logistique, baseline dummy.
    - Les importances LightGBM sont stabilisées sur plusieurs folds et plusieurs seeds.
    - Le modèle final est tuné avec `RandomizedSearchCV` sur le coût métier.
    - Le seuil de décision est optimisé sur calibration.
