# Application Prédiction

<div style="padding: 1rem 1.25rem; border-left: 0.28rem solid #448aff; background: rgba(68, 138, 255, 0.10); border-radius: 0.25rem; font-size: 1.08rem; line-height: 1.5;">
L'application Streamlit rend le modèle <strong>manipulable par un utilisateur métier</strong> : charger, modifier, prédire, comparer.
</div>

## Parcours utilisateur

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)

```mermaid
%%{init: {"themeVariables": {"fontSize": "35px"}, "flowchart": {"nodeSpacing": 60, "rankSpacing": 80, "arrowMarkerAbsolute": true}} }%%
flowchart LR
    A[Saisir ID client] --> B[Lookup API]
    B --> C[Features préremplies]
    C --> D[Edition manuelle]
    D --> E[Predict]
    E --> F[Jauge + verdict]
```

## Ce que l'app apporte

- Les 20 features sont regroupées par **thème métier**.
- Les noms techniques sont remplacés par des labels plus lisibles.
- Les mini-distributions situent le client face à la référence.
- La jauge montre la probabilité, le seuil et la décision.

## Lien avec l'API

| Endpoint | Usage dans l'app |
| --- | --- |
| `/lookup/{sk_id}` | Charger un client |
| `/reference` | Comparer aux distributions |
| `/model-info` | Afficher le seuil |
| `/predict` | Calculer la décision |

## Démo

!!! tip "Démo à ouvrir"
    Lancer l'application avec **`just dashboard`**, puis ouvrir :

    - **Application Streamlit - Prédiction** : [http://localhost:8501](http://localhost:8501)

    --> chargement d'un client, modification d'une feature, nouvelle prédiction.

??? info "Annexes"

    ## Affichage détaillé

    - Les features sont organisées par thèmes : scores externes, profil, prêt, remboursement, crédit, carte bancaire.
    - Les catégories sont décodées pour afficher genre et niveau d'études en clair.
    - Les valeurs sont reformatées avant l'envoi API : entiers, flottants et valeurs manquantes.
    - La session Streamlit conserve les features éditées entre deux actions.

    ## Comportement API côté app

    - `/lookup/{sk_id}` préremplit les données client.
    - `/reference` alimente les mini-distributions de comparaison.
    - `/model-info` fournit le seuil de décision.
    - `/predict` recalcule le score après édition.
    - Le cache Streamlit évite de recharger seuil, référence et graphes à chaque action.
