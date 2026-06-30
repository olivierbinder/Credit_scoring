# Optimisation de l'inférence

## Objectif

L'objectif est d'améliorer le temps de réponse du modèle en production.

Après déploiement, on mesure les performances réelles ou simulées :

- temps d'inférence ;
- latence moyenne ;
- P95 et P99 ;
- consommation CPU ;
- consommation mémoire.

## Profiling

Un profiling CPU est réalisé sur plusieurs inférences successives.

Le rapport de profiling identifie les fonctions les plus coûteuses :

- nombre d'appels ;
- temps propre ;
- temps cumulé.

Cela permet d'observer le coût du chemin d'inférence complet, pas seulement le calcul du modèle.

## Goulots d'étranglement observés

Le profiling montre surtout des coûts liés à l'environnement Python :

- conversions `pandas` ;
- validation et accès dynamiques ;
- appels internes de librairies ;
- mesure de métriques système via `psutil` ;
- surcoût potentiel lié au wrapping du modèle.

Ces éléments ne sont pas forcément coûteux individuellement, mais ils peuvent s'accumuler sur un endpoint de prédiction.

## Optimisation testée : ONNX Runtime

La stratégie principale testée consiste à convertir le modèle en **ONNX** puis à exécuter l'inférence avec **ONNX Runtime**.

L'intérêt est de réduire le surcoût Python et d'exécuter un graphe optimisé.

## Benchmark

Le benchmark compare :

- modèle LightGBM standard ;
- modèle LightGBM converti en ONNX.

Les métriques mesurées sont :

| Métrique | Description |
|---|---|
| Latence moyenne | Temps moyen d'inférence |
| P95 | 95 % des inférences sont plus rapides que cette valeur |
| P99 | 99 % des inférences sont plus rapides que cette valeur |
| Speedup | Rapport entre la latence LightGBM et la latence ONNX |

## Résultat attendu

Le benchmark doit montrer une réduction concrète du temps d'inférence.

Exemple de formulation en soutenance :

> La version ONNX réduit fortement le temps d'inférence en exécutant un graphe optimisé avec ONNX Runtime. Le gain mesuré est de X fois par rapport au modèle standard.

## Intégration dans le dépôt

L'optimisation est intégrée dans le code de serving :

- chargement de la session ONNX ;
- préparation des features au format `float32` ;
- appel à `session.run(...)` ;
- conservation de la même logique de seuil et de décision.

## Conclusion optimisation

L'optimisation ONNX permet de réduire la latence d'inférence sans changer l'interface de l'API.
