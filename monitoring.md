# Credit Scoring API — Production Monitoring

Architecture PoC entièrement locale (zéro cloud requis).

---

## Architecture

```
┌─────────────────────────────────────┐
│          FastAPI  (api.py)          │
│                                     │
│  Middleware HTTP  ──► api_calls.jsonl
│  POST /predict   ──► predictions.jsonl
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   drift_analysis.py  (CLI script)   │
│                                     │
│  1. Operational KPIs (latence,      │
│     error rate)                     │
│  2. Prediction drift                │
│  3. Feature drift — Evidently AI    │
│  4. Missing value rates             │
│                                     │
│  ──► reports/drift_report.html      │
│  ──► reports/analysis_summary.json  │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│     dashboard.py  (Streamlit)       │
│                                     │
│  Tab 1 — Operational health         │
│  Tab 2 — Prediction drift           │
│  Tab 3 — Feature drift (Evidently)  │
│  Tab 4 — Missing values             │
│  Tab 5 — Log explorer               │
└─────────────────────────────────────┘
```

---

## Données loggées

### `logs/predictions.jsonl` (une ligne JSON par prédiction)

| Champ | Description |
|---|---|
| `request_id` | UUID de corrélation |
| `timestamp` | ISO 8601 UTC |
| `inputs.*` | Toutes les features brutes (avant encodage) |
| `probability` | Score de défaut [0, 1] |
| `prediction_label` | Label texte |
| `latency_ms` | Temps d'inférence |
| `success` | `true` / `false` |
| `error` | Message d'erreur si `success=false` |

### `logs/api_calls.jsonl` (une ligne JSON par appel HTTP)

| Champ | Description |
|---|---|
| `request_id` | UUID (corrélable avec predictions) |
| `method` | GET / POST |
| `path` | Route appelée |
| `status_code` | Code HTTP |
| `latency_ms` | Latence totale requête |
| `is_error` | `true` si status >= 400 |

---

## Installation

```bash
pip install evidently rich streamlit fastapi uvicorn
```

---

## Utilisation

### 1. Démarrer l'API

```bash
uvicorn credit_scoring.serving.api:app --reload
```

### 2. Générer des logs de test (sans API live)

```bash
python generate_sample_logs.py --n 3000 --output logs/
```

### 3. Analyse automatique (CLI)

```bash
python drift_analysis.py \
    --predictions logs/predictions.jsonl \
    --api-calls   logs/api_calls.jsonl   \
    --reference   data/reference.parquet \
    --output      reports/
```

Produit :
- `reports/drift_report.html` — rapport Evidently interactif
- `reports/analysis_summary.json` — résumé JSON de toutes les métriques

### 4. Dashboard Streamlit

```bash
streamlit run dashboard.py -- \
    --predictions logs/predictions.jsonl \
    --api-calls   logs/api_calls.jsonl   \
    --reference   data/reference.parquet
```

---

## Seuils d'alerte

| Métrique | Seuil | Niveau |
|---|---|---|
| Error rate | > 5 % | 🔴 CRITICAL |
| P95 Latency | > 2 000 ms | 🔴 CRITICAL |
| Feature drift share | > 30 % des features | 🔴 CRITICAL |
| Feature individuelle driftée | p-value < 0.05 | ⚠️ WARNING |
| Missing rate feature nullable | > 80 % | ⚠️ WARNING |
| Probability delta (baseline vs récent) | > 0.10 | 🔴 CRITICAL |

---

## Détection de drift — Points de vigilance

### Features critiques à surveiller en priorité

**EXT_SOURCE_1/2/3** — scores externes tiers. Une dégradation du partenaire de scoring
se traduit immédiatement par un shift de distribution et une dégradation des prédictions.

**DAYS_EMPLOYED** — fortement corrélé au risque. Sensible aux cycles économiques
(hausse du chômage → shift vers 0 ou valeurs manquantes).

**INSTAL_DPD_MEAN** — indicateur retardé : le drift arrive 1–3 mois après un choc économique.

**AMT_ANNUITY / AMT_GOODS_PRICE** — sensibles à l'inflation. Un drift continu
sans recalibration dégrade la précision du `PAYMENT_RATE` dérivé.

### Référence de drift

La référence est construite depuis les données d'entraînement (parquet).
Fenêtre de comparaison recommandée : **30 jours glissants** en production.

### Tests statistiques utilisés par Evidently

| Type de feature | Test par défaut | Seuil p-value |
|---|---|---|
| Numérique (n > 1000) | Wasserstein distance | 0.1 |
| Numérique (n ≤ 1000) | Kolmogorov-Smirnov | 0.05 |
| Catégorielle | Chi-² | 0.05 |

### RGPD

- Aucune donnée personnelle directement identifiable n'est loggée
  (`SK_ID_CURR` non inclus dans les logs de prédiction par défaut).
- Les features loggées sont des agrégats / scores — pas de nom, adresse, NIR.
- Durée de rétention recommandée : 12 mois (à définir dans votre DPA).
- Les logs `.jsonl` doivent être stockés sur un volume chiffré en production.

---

## Intégration CI/CD suggérée

```yaml
# .github/workflows/drift_check.yml
- name: Run drift analysis
  run: |
    python drift_analysis.py \
      --predictions logs/predictions.jsonl \
      --api-calls   logs/api_calls.jsonl   \
      --reference   data/reference.parquet \
      --output      reports/
    # Exit code non-zero si alertes critiques → bloque le déploiement
```