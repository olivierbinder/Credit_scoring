# Contexte de la mission

## Objectif

L'objectif du projet est de mettre en production un modèle de **credit scoring** capable d'estimer le risque de défaut d'un client à partir de données de demande de crédit, d'historique de remboursement et d'informations externes.

La soutenance se concentre principalement sur la partie **MLOps** :

- exposition du modèle via une API ;
- déploiement Docker sur Hugging Face Spaces ;
- mise en place d'un pipeline CI/CD ;
- monitoring de l'API et des données ;
- optimisation du temps d'inférence.

## Périmètre fonctionnel

Le modèle retourne :

- une probabilité de défaut ;
- une décision binaire selon un seuil optimal ;
- une réponse exploitable par une application ou un service métier.

Exemple de sortie attendue :

```json
{
  "probability": 0.1842,
  "prediction": "Not likely to default"
}
```

## Architecture générale

Le projet est organisé autour de quatre blocs :

1. **API FastAPI** : expose les routes `/predict`, `/lookup/{sk_id}`, `/model-info` et `/reference`.
2. **Application Streamlit** : permet de piloter le monitoring, la dérive des données et les benchmarks d'inférence.
3. **Monitoring** : collecte des logs API, logs de prédiction, rapports Evidently et métriques de performance.
4. **CI/CD** : automatisation des tests, construction Docker et déploiement sur Hugging Face Spaces.

## Points à démontrer

Pendant la soutenance, je montre :

- une requête réelle vers l'API déployée ;
- la réponse du modèle avec le score prédit ;
- les rapports de monitoring ;
- les métriques de performance ;
- le pipeline CI/CD déclenché par un commit Git.
