# Structure du dépôt GitHub

## Objectif

Cette partie sert à montrer rapidement l'organisation du code et les fichiers importants pour le déploiement MLOps.

## Structure générale

```text
credit-scoring/
├── src/
│   └── credit_scoring/
│       ├── config.py
│       ├── serving/
│       │   ├── api.py
│       │   └── inference.py
│       ├── preprocessing/
│       ├── training/
│       └── monitoring/
├── reports/
│   ├── drift_report.html
│   └── quality_report.html
├── tests/
├── docs/
├── Dockerfile
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
```

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `src/credit_scoring/serving/api.py` | API FastAPI |
| `src/credit_scoring/serving/inference.py` | Chargement modèle et prédiction |
| `src/credit_scoring/config.py` | Chemins, mappings, constantes |
| `Dockerfile` | Image de déploiement |
| `pyproject.toml` | Dépendances Python et groupes dev/ci |
| `.github/workflows/...` | Pipeline CI/CD |
| `README.md` | Documentation principale du projet |
| `reports/` | Rapports générés par le monitoring |

## Dockerfile

Le Dockerfile permet de construire une image reproductible contenant :

- l'application ;
- les dépendances ;
- le modèle ;
- la commande de lancement.

## Configuration Python

Le projet utilise :

- Python 3.12 ;
- `uv` pour la gestion des dépendances ;
- `pytest` pour les tests ;
- `ruff` pour la qualité du code.

## Documentation

La documentation Zensical est stockée dans le dossier `docs/`.

Elle permet de présenter :

- le contexte ;
- le monitoring ;
- les optimisations ;
- la démo API ;
- la CI/CD ;
- la structure du dépôt.
