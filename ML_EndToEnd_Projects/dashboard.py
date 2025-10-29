import streamlit as st
import requests

st.set_page_config(page_title="ML & GenAI Project Suite", layout="wide")
st.title("ML & GenAI Project Suite Demo Dashboard")

st.markdown("""
Welcome! Launch and interact with any of the 15 end-to-end ML and GenAI projects below. Each project exposes a FastAPI API and can be demoed live. For a visual experience, use the Streamlit dashboards where available.
""")

projects = [
    {"name": "E-commerce Recommender", "folder": "traditional_ml/01_ecommerce_recommender", "api": "http://localhost:8000/docs"},
    {"name": "Credit Risk Assessment", "folder": "traditional_ml/02_credit_risk_assessment", "api": "http://localhost:8000/docs"},
    {"name": "Stock Forecasting", "folder": "traditional_ml/03_stock_forecasting", "api": "http://localhost:8000/docs"},
    {"name": "Computer Vision System", "folder": "traditional_ml/04_computer_vision", "api": "http://localhost:8000/docs"},
    {"name": "NLP Text Analysis", "folder": "traditional_ml/05_nlp_text_analysis", "api": "http://localhost:8000/docs"},
    {"name": "Fraud Detection", "folder": "traditional_ml/06_fraud_detection", "api": "http://localhost:8000/docs"},
    {"name": "Customer Churn", "folder": "traditional_ml/07_customer_churn", "api": "http://localhost:8000/docs"},
    {"name": "Supply Chain Optimization", "folder": "traditional_ml/08_supply_chain_optimization", "api": "http://localhost:8000/docs"},
    {"name": "Energy Prediction", "folder": "traditional_ml/09_energy_prediction", "api": "http://localhost:8000/docs"},
    {"name": "Autonomous Vehicle System", "folder": "traditional_ml/10_autonomous_vehicle", "api": "http://localhost:8000/docs"},
    {"name": "AI Code Generator", "folder": "gen_ai/11_ai_code_generator", "api": "http://localhost:8000/docs"},
    {"name": "AI Content Creator", "folder": "gen_ai/12_ai_content_creator", "api": "http://localhost:8000/docs"},
    {"name": "Document Intelligence", "folder": "gen_ai/13_document_intelligence", "api": "http://localhost:8000/docs"},
    {"name": "Conversational AI", "folder": "gen_ai/14_conversational_ai", "api": "http://localhost:8000/docs"},
    {"name": "Drug Discovery AI", "folder": "gen_ai/15_drug_discovery_ai", "api": "http://localhost:8000/docs"},
]

cols = st.columns(3)
for i, proj in enumerate(projects):
    with cols[i % 3]:
        st.subheader(proj["name"])
        st.write(f"**Folder:** `{proj['folder']}`")
        st.write(f"[Open API Docs]({proj['api']})")
        st.write(f"Launch: `cd {proj['folder']} && uvicorn app:app --reload`")
        st.markdown("---")

st.info("To run a project, open a terminal, navigate to the folder, and launch the FastAPI app. Then use the API docs or build a Streamlit dashboard for a visual demo.")
