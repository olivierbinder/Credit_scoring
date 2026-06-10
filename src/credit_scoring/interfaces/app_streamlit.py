import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

st.title("📊 Credit Scoring Predictor")
st.markdown("Ajustez les paramètres du client pour obtenir une prédiction de risque.")

# Définition des features avec des labels plus lisibles
features_config = {
    "EXT_SOURCE_3": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_1": 0.5,
    "PAYMENT_RATE": 0.05,
    "DAYS_EMPLOYED": 1500.0,
    "AMT_ANNUITY": 25000.0,
    "AMT_GOODS_PRICE": 500000.0,
    "DAYS_BIRTH": -15000.0,
    "CODE_GENDER": 0.0,
    "INSTAL_DPD_MEAN": 0.0,
    "OWN_CAR_AGE": 5.0,
    "AMT_CREDIT": 500000.0,
    "DAYS_ID_PUBLISH": -2000.0,
    "INSTAL_AMT_PAYMENT_SUM": 100000.0,
    "POS_CNT_INSTALMENT_FUTURE_MEAN": 10.0,
    "NAME_EDUCATION_TYPE": 2.0,
    "ANNUITY_INCOME_PERC": 0.1,
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN": -500.0,
    "REGION_POPULATION_RELATIVE": 0.02,
    "DAYS_EMPLOYED_PERC": 0.1,
}

# Utilisation de colonnes pour organiser l'interface (4 colonnes)
cols = st.columns(4)
user_inputs = {}

for i, (name, default) in enumerate(features_config.items()):
    with cols[i % 4]:
        user_inputs[name] = st.number_input(name, value=default, format="%.4f")

if st.button("Prédire le risque"):
    try:
        response = requests.post(API_URL, json=user_inputs)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            prob = result["probability"]

            # Affichage visuel du résultat
            st.subheader("Résultat")
            color = "red" if prediction == 1 else "green"
            st.metric(label="Probabilité de risque", value=f"{prob:.2%}")
            st.markdown(
                f"### Classification finale : :{color}[{'À Risque' if prediction == 1 else 'Approuvé'}]"
            )
        else:
            st.error(f"Erreur API : {response.text}")
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
