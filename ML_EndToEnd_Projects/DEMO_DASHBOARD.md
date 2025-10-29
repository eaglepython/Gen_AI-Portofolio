# ML & GenAI Demo Dashboard

Welcome! This is your interactive demo dashboard for all 15 end-to-end ML and GenAI projects. Each project exposes a FastAPI web interface for live demo and presentation.

---

## 🚀 Launch Any Demo

| Project Name                | Launch Command                                      | API Docs URL                  |
|----------------------------|-----------------------------------------------------|-------------------------------|
| E-commerce Recommender     | `cd traditional_ml/01_ecommerce_recommender && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Credit Risk Assessment     | `cd traditional_ml/02_credit_risk_assessment && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Stock Forecasting          | `cd traditional_ml/03_stock_forecasting && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Computer Vision System     | `cd traditional_ml/04_computer_vision && uvicorn app:app --reload` | http://localhost:8000/docs    |
| NLP Text Analysis          | `cd traditional_ml/05_nlp_text_analysis && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Fraud Detection            | `cd traditional_ml/06_fraud_detection && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Customer Churn             | `cd traditional_ml/07_customer_churn && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Supply Chain Optimization  | `cd traditional_ml/08_supply_chain_optimization && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Energy Prediction          | `cd traditional_ml/09_energy_prediction && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Autonomous Vehicle System  | `cd traditional_ml/10_autonomous_vehicle && uvicorn app:app --reload` | http://localhost:8000/docs    |
| AI Code Generator          | `cd gen_ai/11_ai_code_generator && uvicorn app:app --reload` | http://localhost:8000/docs    |
| AI Content Creator         | `cd gen_ai/12_ai_content_creator && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Document Intelligence      | `cd gen_ai/13_document_intelligence && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Conversational AI          | `cd gen_ai/14_conversational_ai && uvicorn app:app --reload` | http://localhost:8000/docs    |
| Drug Discovery AI          | `cd gen_ai/15_drug_discovery_ai && uvicorn app:app --reload` | http://localhost:8000/docs    |

---

## 🖥️ How to Demo

1. Open a terminal and navigate to the project folder.
2. Run the launch command for the project you want to demo.
3. Open your browser to `http://localhost:8000/docs` for the interactive API interface.
4. Present live predictions, results, and model outputs directly from the web UI.

---

## 📊 Optional: Streamlit Dashboards
- For a more visual, interactive demo, you can add a `dashboard.py` using Streamlit to any project.
- Launch with: `streamlit run dashboard.py`

---

## 🧹 Repo Organization
- Only folders with working code and FastAPI apps are kept.
- All demo commands and docs are up to date for presentation.

---

Enjoy your live ML & GenAI demo suite!
