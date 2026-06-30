# Déroulé de démonstration

## 1. Introduire le projet

- Problème métier : prédire un risque de défaut.
- Objectif MLOps : passer d'un modèle à une application déployée.
- Périmètre : modèle final, API, interface, monitoring, CI/CD.

Phrase utile :

> Le projet ne présente pas seulement un score, mais une mini-chaîne de mise en
> production : servir, observer et redéployer le modèle.

## 2. Conception du modèle

- Montrer les sources de données.
- Expliquer les agrégations historiques au niveau client.
- Expliquer la réduction à 20 features.
- Mentionner MLflow et le seuil optimisé.

## 3. API FastAPI

- Ouvrir Swagger sur `/docs`.
- Montrer `/model-info`.
- Montrer `/lookup/{sk_id}`.
- Tester ou expliquer `/predict`.
- Montrer le format d'erreur avec `request_id`.

## 4. Application Prédiction

- Charger un client.
- Montrer les groupes de features.
- Modifier une valeur simple.
- Montrer les distributions de référence.
- Lancer la prédiction et commenter la jauge.

## 5. Application Monitoring

- Onglet API :
  - routes métier uniquement ;
  - volume, erreurs et latence.
- Onglet dérive :
  - rapport Evidently ;
  - qualité des données.
- Onglet optimisation :
  - profiling ;
  - benchmark LightGBM vs ONNX.

## 6. CI/CD et déploiement

- Montrer `ci.yml` : Ruff et Pytest.
- Montrer `cd.yml` : déclenchement après CI verte sur `main`.
- Montrer le checkout Git LFS.
- Montrer le push vers Hugging Face Spaces.
- Ouvrir le Space déployé.

## 7. Conclusion

- Le modèle est utilisable via API et interface.
- L'application permet de tester des scénarios client.
- Le monitoring donne une visibilité sur API, données et performance.
- Le déploiement est automatisé après validation.

Message final :

> La valeur du projet est dans la chaîne complète : modèle, service, interface,
> observabilité et livraison automatisée.
