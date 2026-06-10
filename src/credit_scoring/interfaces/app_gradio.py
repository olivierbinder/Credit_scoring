import gradio as gr
import requests

# L'URL où tourne votre API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

feature_names = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "PAYMENT_RATE",
    "DAYS_EMPLOYED",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "CODE_GENDER",
    "INSTAL_DPD_MEAN",
    "OWN_CAR_AGE",
    "AMT_CREDIT",
    "DAYS_ID_PUBLISH",
    "INSTAL_AMT_PAYMENT_SUM",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
    "NAME_EDUCATION_TYPE",
    "ANNUITY_INCOME_PERC",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN",
    "REGION_POPULATION_RELATIVE",
    "DAYS_EMPLOYED_PERC",
]

default_values = [
    0.5,  # EXT_SOURCE_3 (Source de données externe 3 - normalisé)
    0.6,  # EXT_SOURCE_2 (Source de données externe 2 - normalisé)
    0.5,  # EXT_SOURCE_1 (Source de données externe 1 - normalisé)
    0.05,  # PAYMENT_RATE (Ratio de paiement ~5%)
    1500.0,  # DAYS_EMPLOYED (Environ 4 ans d'ancienneté)
    25000.0,  # AMT_ANNUITY (Annuité annuelle)
    500000.0,  # AMT_GOODS_PRICE (Valeur du bien)
    -15000.0,  # DAYS_BIRTH (Âge ~41 ans : -15000 jours)
    0.0,  # CODE_GENDER (0 pour Homme, 1 pour Femme, par ex.)
    0.0,  # INSTAL_DPD_MEAN (Jours de retard de paiement : 0)
    5.0,  # OWN_CAR_AGE (Âge de la voiture ~5 ans)
    500000.0,  # AMT_CREDIT (Montant du crédit)
    -2000.0,  # DAYS_ID_PUBLISH (Date de publication de l'ID)
    100000.0,  # INSTAL_AMT_PAYMENT_SUM (Total des paiements)
    10.0,  # POS_CNT_INSTALMENT_FUTURE_MEAN (Échéances futures)
    2.0,  # NAME_EDUCATION_TYPE (Niveau d'éducation)
    0.1,  # ANNUITY_INCOME_PERC (Ratio annuité/revenu ~10%)
    -500.0,  # PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN
    0.02,  # REGION_POPULATION_RELATIVE (Densité population)
    0.1,  # DAYS_EMPLOYED_PERC (Ratio ancienneté/âge)
]


def predict_score(*args):
    payload = dict(zip(feature_names, args))
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        if response.status_code == 200:
            return f"Classe : {result['prediction']} | Probabilité de risque : {result['probability']:.2%}"
        else:
            return f"Erreur : {result.get('detail', 'Inconnue')}"
    except Exception as e:
        return f"Erreur de connexion : {str(e)}"


# Création des entrées avec valeurs par défaut
inputs = [
    gr.Number(label=name, value=val) for name, val in zip(feature_names, default_values)
]

demo = gr.Interface(
    fn=predict_score, inputs=inputs, outputs="text", title="Credit Scoring Demo"
)

if __name__ == "__main__":
    demo.launch()
