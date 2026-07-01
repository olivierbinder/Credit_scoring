# Contexte de la mission

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
Transformer un <strong>modèle de prédiction</strong> de risque de défaut d'un demandeur de crédit en <strong>application exploitable</strong>.
</div>

## Schéma général

```mermaid
%%{init: {"themeVariables": {"fontSize": "40px"}, "flowchart": {"nodeSpacing": 60, "rankSpacing": 80, "arrowMarkerAbsolute": true}} }%%
flowchart LR
    A[Données Home Credit] --> B[Préparation données]
    B --> C[Modèle LightGBM]
    C --> D[API FastAPI]
    D --> E[App Streamlit]
    D --> F[Logs + monitoring]
    E --> G[Déploiement HF Spaces]
    F --> G
```


## Stack MLOps

[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
