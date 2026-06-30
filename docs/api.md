# Démonstration de l'API déployée

## Objectif

Cette partie démontre que le modèle est bien exposé via une API et qu'il peut être appelé comme un service de prédiction.

## Routes principales

| Route | Méthode | Rôle |
|---|---:|---|
| `/` | GET | Health check |
| `/model-info` | GET | Retourne le seuil du modèle |
| `/lookup/{sk_id}` | GET | Récupère les features d'un client |
| `/predict` | POST | Retourne le score de défaut |

## Health check

```bash
curl https://<url-space-huggingface>/
```

Réponse attendue :

```json
{
  "status": "ok"
}
```

## Informations modèle

```bash
curl https://<url-space-huggingface>/model-info
```

Réponse attendue :

```json
{
  "threshold": 0.42
}
```

## Requête de prédiction

Exemple de requête :

```bash
curl -X POST "https://<url-space-huggingface>/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_1": 0.31,
    "EXT_SOURCE_2": 0.62,
    "EXT_SOURCE_3": 0.45,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 450000,
    "DAYS_BIRTH": -15000,
    "DAYS_EMPLOYED": -1200,
    "PAYMENT_RATE": 0.05,
    "OWN_CAR_AGE": null,
    "CODE_GENDER": "M",
    "NAME_EDUCATION_TYPE": "Higher education",
    "INSTAL_DPD_MEAN": 0,
    "INSTAL_AMT_PAYMENT_SUM": 750000,
    "POS_CNT_INSTALMENT_FUTURE_MEAN": 8,
    "POS_SK_DPD_DEF_MEAN": 0,
    "PREV_CNT_PAYMENT_MEAN": 12,
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN": -500,
    "ACTIVE_DAYS_CREDIT_MAX": -120,
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0,
    "CC_CNT_DRAWINGS_CURRENT_VAR": 0
  }'
```

Réponse attendue :

```json
{
  "probability": 0.1842,
  "prediction": "Not likely to default"
}
```

## Points à montrer en soutenance

Pendant la démo :

1. ouvrir l'URL de l'API ;
2. montrer le health check ;
3. envoyer une requête `/predict` ;
4. montrer la réponse JSON ;
5. vérifier que l'appel apparaît ensuite dans les logs de monitoring.
