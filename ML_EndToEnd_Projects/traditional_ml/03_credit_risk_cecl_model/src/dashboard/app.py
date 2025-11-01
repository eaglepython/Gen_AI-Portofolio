import streamlit as st
import pandas as pd
import requests
import logging
import os

st.set_page_config(page_title="Credit Risk & CECL Dashboard", layout="wide")
st.title("Credit Risk & CECL Model Dashboard")

st.sidebar.header("Input Features")
features = st.sidebar.text_area("Enter features as comma-separated values", "0.5, 100000, 5, 1, 0")
features_list = [float(x.strip()) for x in features.split(",") if x.strip()]

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'dashboard.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

if st.sidebar.button("Score Models"):
    payload = {"features": features_list}
    try:
        pd_response = requests.post("http://localhost:8000/score/pd", json=payload)
        lgd_response = requests.post("http://localhost:8000/score/lgd", json=payload)
        ead_response = requests.post("http://localhost:8000/score/ead", json=payload)
        pd_score = pd_response.json().get("pd")
        lgd_score = lgd_response.json().get("lgd")
        ead_score = ead_response.json().get("ead")
        st.subheader("Model Scores")
        st.write({
            "PD": pd_score,
            "LGD": lgd_score,
            "EAD": ead_score
        })
        logging.info(f"Input: {features_list} | PD: {pd_score} | LGD: {lgd_score} | EAD: {ead_score}")
    except Exception as e:
        st.error(f"Error scoring models: {e}")
        logging.error(f"Error scoring models for input {features_list}: {e}")

st.markdown("---")
st.markdown("### Model Monitoring & Validation (Placeholder)")
st.write("Add charts, validation metrics, and monitoring here.")
