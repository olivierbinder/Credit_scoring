# Structure du dépôt

## Objectif

Montrer que le projet est organisé comme une application ML maintenable :

- code métier dans `src/credit_scoring` ;
- jobs ponctuels dans `scripts` ;
- configuration dans `config` ;
- documentation dans `docs` ;
- workflows dans `.github/workflows`.

## Vue générale

```text
credit-scoring/
|-- config/
|   `-- training.yaml
|-- docs/
|-- scripts/
|-- src/credit_scoring/
|   |-- features/
|   |-- models/
|   |-- serving/
|   `-- interfaces/
|-- tests/
|-- Dockerfile
|-- Justfile
|-- pyproject.toml
|-- zensical.toml
`-- .github/workflows/
```

## Dossiers principaux

| Dossier | Rôle |
|---|---|
| `features/` | Préprocessing et sélection de features |
| `models/` | Training, tuning, évaluation, explainability |
| `serving/` | API, inference, modèle packagé, fichiers de référence |
| `interfaces/` | Application Streamlit multi-pages |
| `scripts/` | Jobs ponctuels : ONNX, base inference, feature selection |
| `docs/` | Support Zensical de soutenance |

## Fichiers clés

- `config/training.yaml` : modèle choisi, paramètres et options training.
- `src/credit_scoring/config.py` : chemins, features finales et mappings.
- `src/credit_scoring/serving/api.py` : routes FastAPI et logging.
- `src/credit_scoring/serving/inference.py` : chargement modèle et prédiction.
- `src/credit_scoring/interfaces/app_streamlit.py` : entrée Streamlit.
- `Dockerfile` : image de déploiement.
- `Justfile` : commandes locales utiles.
- `.github/workflows/ci.yml` : lint et tests.
- `.github/workflows/cd.yml` : déploiement Hugging Face.

## Justfile

Le `Justfile` sert de façade simple pour les commandes courantes :

- `just api` ;
- `just dashboard` ;
- `just app` ;
- `just train` ;
- `just export-onnx` ;
- `just test`.

## Message clé

Le dépôt sépare la logique réutilisable du package Python et les scripts
d'orchestration ponctuels. Cette séparation rend le projet plus lisible pour la
soutenance et plus maintenable.
