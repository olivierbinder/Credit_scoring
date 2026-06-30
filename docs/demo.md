# Déroulé de démonstration pour la soutenance

## 1. Présenter le contexte

Expliquer rapidement :

- le problème métier : prédire le risque de défaut ;
- le modèle utilisé ;
- le passage d'un notebook à un service déployé ;
- l'objectif MLOps : rendre le modèle exploitable, monitoré et déployable.

## 2. Montrer l'application Streamlit

Dans Streamlit, parcourir les trois onglets principaux :

1. **Supervision de l'API** : latence, erreurs, volume de requêtes.
2. **Surveillance de la dérive** : rapports Evidently.
3. **Optimisation de l'inférence** : profiling et benchmark ONNX.

## 3. Démontrer l'API

Faire un appel à `/predict` avec une requête JSON.

Montrer :

- le score de défaut ;
- la décision associée ;
- le fait que l'appel est ensuite journalisé dans les logs.

## 4. Montrer le monitoring

Afficher :

- le rapport de dérive ;
- le rapport qualité ;
- les logs API ;
- les métriques de latence et d'erreurs.

Phrase possible :

> Le monitoring permet de vérifier que les données en production restent proches des données de référence et que l'API reste stable en termes de latence et d'erreurs.

## 5. Montrer l'optimisation

Afficher :

- le profiling ;
- le benchmark LightGBM vs ONNX ;
- le gain sur la latence moyenne ;
- le P95 et le P99.

Phrase possible :

> Le profiling met en évidence le coût du chemin d'inférence complet. La conversion ONNX permet de réduire le surcoût Python et d'améliorer concrètement la latence.

## 6. Montrer le dépôt GitHub

Naviguer rapidement dans :

- `src/credit_scoring/serving/api.py` ;
- `src/credit_scoring/serving/inference.py` ;
- `Dockerfile` ;
- `.github/workflows/` ;
- `pyproject.toml` ;
- `README.md`.

## 7. Montrer la CI/CD

Faire ou montrer un commit récent.

Puis afficher :

- le déclenchement GitHub Actions ;
- les tests ;
- la construction Docker ;
- le déploiement Hugging Face Spaces.

## Conclusion

Message final :

> Le projet couvre une chaîne MLOps complète : modèle exposé via API, déploiement Docker, monitoring des données et de l'API, optimisation de l'inférence, et CI/CD pour automatiser la livraison.
