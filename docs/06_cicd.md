# CI/CD et Déploiement

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
La chaîne CI/CD garantit que l'application est <strong>testée, packagée et déployée de façon reproductible</strong>.
</div>

## Workflow

[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Ruff](https://img.shields.io/badge/Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

```mermaid
%%{init: {"themeVariables": {"fontSize": "40px"}, "flowchart": {"nodeSpacing": 60, "rankSpacing": 80, "arrowMarkerAbsolute": true}} }%%
flowchart LR
    A[Push] --> B[CI]
    B --> C[Ruff]
    B --> D[Pytest]
    C --> E{CI verte}
    D --> E
    E -->|main| F[CD]
    F --> G[Git LFS pull]
    G --> H[Push HF Space]
    H --> I[Build Docker]
```


| Brique | Rôle |
| --- | --- |
| Ruff + Pytest | Bloquer les régressions |
| Dockerfile | Regrouper API, Streamlit, modèle et DB |
| Git LFS | Fournir les artefacts lourds au déploiement |
| Hugging Face Spaces | Héberger l'application publique |

## Docker local

- `Dockerfile` lance API FastAPI + interface Streamlit dans le même conteneur.
- `docker-compose.yml` sert au test local complet avant déploiement.

## Démo

!!! tip "Démo à ouvrir"
    Ouvrir les pages externes :

    - **Runs GitHub Actions** : [https://github.com/olivierbinder/Credit_scoring/actions](https://github.com/olivierbinder/Credit_scoring/actions)
    - **Space Hugging Face** : [https://huggingface.co/spaces/Benderrrrr/credit-scoring](https://huggingface.co/spaces/Benderrrrr/credit-scoring)

??? info "Annexes"

    ## CI

    - La CI se lance à chaque `push`.
    - GitHub Actions installe `uv` avec cache activé.
    - Ruff vérifie le code avec `ruff check src/ tests/`.
    - Le formatage est contrôlé avec `ruff format --check src/ tests/`.
    - Pytest valide les tests unitaires et API.

    ## CD et Docker

    - La CD attend une CI verte sur `main`.
    - `workflow_dispatch` permet un redéploiement manuel.
    - Le Dockerfile part de `python:3.12-slim`.
    - Les ports `8000` et `7860` exposent FastAPI et Streamlit.
    - Hugging Face reconstruit l'image après le push vers le Space.

    ## Git LFS

    - Les données locales et caches restent exclus du dépôt.
    - `.gitattributes` redirige les artefacts lourds vers LFS : `*.parquet`, `*.pkl`, `*.onnx`.
    - La CD exécute `git lfs pull` pour récupérer les vrais fichiers lourds.
