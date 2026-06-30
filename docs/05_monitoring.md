# Application Monitoring

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
Le monitoring répond à trois questions : <strong>l'API répond-elle ? les données dérivent-elles ? l'inférence coûte-t-elle cher ?</strong>
</div>

## 1. Supervision de l'API

[![psutil](https://img.shields.io/badge/psutil-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://psutil.readthedocs.io/)

```mermaid
flowchart LR
    A[API FastAPI] --> B[api_calls.jsonl]
    A --> C[predictions.jsonl]
    B --> D[Volume / erreurs / latence]
    C --> E[Inférence / CPU / mémoire]
    D --> F[Dashboard Streamlit]
    E --> F
```

| Signal | Source |
| --- | --- |
| Latence par route | Middleware FastAPI |
| Volume / erreurs | `logs/api_calls.jsonl` |
| Inference ms | `logs/predictions.jsonl` |
| CPU / mémoire | `psutil` |

Les courbes runtime sont **lissées par moyenne mobile** pour rendre la tendance lisible en démo.

## 2. Dérive et qualité

[![Evidently](https://img.shields.io/badge/Evidently-6C2BD9?style=for-the-badge&logo=evidently&logoColor=white)](https://www.evidentlyai.com/)

- `reference.parquet` sert de population de référence.
- `test.parquet` sert de jeu courant de comparaison.
- Evidently génère les rapports `drift_report.html` et `quality_report.html`.

## 3. Profiling et benchmark

[![cProfile](https://img.shields.io/badge/cProfile-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/profile.html)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

```mermaid
%%{init: {"themeVariables": {"fontSize": "35px"}, "flowchart": {"nodeSpacing": 60, "rankSpacing": 80, "arrowMarkerAbsolute": true}} }%%
flowchart LR
    A[Features client] --> B[DataFrame pandas]
    B --> C[LightGBM]
    B --> D[float32]
    D --> E[ONNX Runtime]
    C --> F[Latence moyenne]
    E --> F
```

- `cProfile` identifie les fonctions Python les plus coûteuses.
- Le benchmark compare LightGBM standard et ONNX sur 100 inférences.
- Les métriques clés sont moyenne, P95, P99, speedup.

## Démo

!!! tip "Démo à ouvrir"
    Lancer l'application avec **`just dashboard`**, puis ouvrir :

    - **Application Streamlit - Pilotage** : [http://localhost:8501](http://localhost:8501)

    --> latence API, coûts runtime, rapport Evidently, benchmark ONNX.

??? info "Annexes"

    ## Supervision API

    | Indicateur | Rôle |
    | --- | --- |
    | Latence moyenne | Temps moyen par route |
    | Nb erreurs | Nombre de réponses HTTP en erreur |
    | Volume | Nombre d'appels |
    | Courbe de latence | Évolution temporelle par route |
    | Temps d'inférence | Coût modèle mesuré sur `/predict` |
    | CPU / mémoire | Ressources runtime collectées avec `psutil` |

    ## Dérive et qualité

    - Les catégories sont décodées avant rapport pour rester lisibles.
    - Evidently utilise Wasserstein pour les numériques et PSI pour les catégorielles.
    - Les rapports HTML sont générés puis affichés dans Streamlit.

    ## Profiling et benchmark

    - Le profiling lance 50 inférences sur un client de référence.
    - `cProfile` mesure les appels Python ; `pstats` trie les fonctions par temps propre.
    - Le benchmark lance 100 inférences par moteur.
    - Un warm-up écarte le coût d'initialisation.
    - Les métriques suivies sont moyenne, P95, P99, speedup et inférences/seconde.
