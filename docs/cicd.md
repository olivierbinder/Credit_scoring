# Pipeline CI/CD

## Objectif

Le pipeline CI/CD automatise la validation et le déploiement de l'application.

L'objectif est de garantir qu'un changement de code :

1. déclenche les tests ;
2. vérifie la qualité du code ;
3. construit l'image Docker ;
4. déploie l'API ou l'application sur Hugging Face Spaces.

## Déclenchement

Le pipeline est déclenché par un événement GitHub, par exemple :

- `push` sur la branche principale ;
- pull request ;
- modification du code applicatif ou des fichiers de configuration.

## Étapes typiques

| Étape | Rôle |
|---|---|
| Checkout | Récupération du dépôt |
| Setup Python | Installation de Python 3.12 |
| uv sync | Installation des dépendances |
| Tests | Exécution de `pytest` |
| Lint | Contrôle du code avec `ruff` |
| Docker build | Construction de l'image |
| Deploy | Déploiement sur Hugging Face Spaces |

## Exemple de commandes

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
docker build -t credit-scoring-api .
```

## Déploiement Hugging Face Spaces

Le déploiement sur Hugging Face Spaces repose sur :

- un `Dockerfile` ;
- les fichiers applicatifs ;
- les dépendances définies dans `pyproject.toml` ;
- le pipeline GitHub Actions.

## Démonstration attendue

Pendant la soutenance :

1. faire une petite modification dans le code ou la documentation ;
2. créer un commit ;
3. pousser sur GitHub ;
4. ouvrir l'onglet Actions ;
5. montrer les jobs : tests, build Docker, déploiement ;
6. ouvrir le Space déployé ;
7. refaire un appel API pour valider le déploiement.

## Message clé

Le pipeline garantit que chaque changement est validé automatiquement avant d'être exposé en production.
