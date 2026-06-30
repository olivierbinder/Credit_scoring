# Monitoring : dérive, logs et performance

## Objectif du monitoring

Le monitoring permet de vérifier que le modèle reste fiable après son déploiement.

Deux dimensions sont suivies :

1. **Dérive des données** : les données reçues en production restent-elles proches des données de référence ?
2. **Fiabilité de l'API** : l'API répond-elle correctement, rapidement et sans erreur ?

## Dérive des données

La dérive des données est analysée avec **Evidently** à partir de deux datasets :

- `PROD_REFERENCE` : données de référence ;
- `PROD_TEST` : données simulées ou observées en production.

Les rapports générés sont :

- `reports/drift_report.html` : rapport de dérive ;
- `reports/quality_report.html` : résumé qualité des données.

## Variables suivies

Le monitoring se concentre sur les 20 variables finales du modèle :

- scores externes ;
- profil du demandeur ;
- caractéristiques du prêt ;
- historique de remboursement ;
- historique de crédit ;
- activité carte de crédit.

Les variables catégorielles encodées sont remappées avant affichage afin de rendre les rapports lisibles :

```python
GENDER_INVERSE = {
    1.0: "M",
    0.0: "F",
}

EDUCATION_INVERSE = {
    0.0: "Lower secondary",
    1.0: "Secondary / secondary special",
    2.0: "Incomplete higher",
    3.0: "Higher education",
    4.0: "Academic degree",
}
```

## Logs API

Chaque appel HTTP est journalisé par un middleware FastAPI.

Les informations collectées sont notamment :

| Champ | Rôle |
|---|---|
| `request_id` | Identifiant unique de requête |
| `timestamp` | Date et heure de l'appel |
| `method` | Méthode HTTP |
| `path` | Route appelée |
| `status_code` | Code de réponse |
| `latency_ms` | Latence de la requête |
| `is_error` | Indicateur d'erreur |
| `client_host` | Adresse du client |

Ces logs alimentent l'onglet **Supervision de l'API** dans Streamlit.

## Logs de prédiction

Lors d'un appel à `/predict`, un second log est produit avec :

- les features d'entrée ;
- la probabilité prédite ;
- le label de prédiction ;
- la latence totale ;
- le temps d'inférence ;
- l'utilisation CPU ;
- l'utilisation mémoire ;
- le statut de succès ou d'erreur.

## Analyse dans Streamlit

L'application Streamlit affiche :

- la performance par route API ;
- le nombre d'erreurs ;
- le volume de requêtes ;
- l'évolution de la latence ;
- les rapports Evidently de dérive et de qualité.

## Conclusion monitoring

Le monitoring permet d'identifier rapidement :

- une dérive des distributions ;
- une hausse des erreurs API ;
- une augmentation de la latence ;
- un problème d'intégrité ou de qualité des données.
