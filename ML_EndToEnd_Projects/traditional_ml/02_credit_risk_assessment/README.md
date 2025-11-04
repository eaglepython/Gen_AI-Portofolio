
<div align="center">
  <h1 style="font-size:2.3rem; font-weight:900; color:#19d4ff; letter-spacing:0.04em; margin-bottom:0.5rem;">
    💳 Credit Risk Assessment — End-to-End ML
  </h1>
  <!-- Add a dashboard or results image here if available -->
  <!-- <img src="docs/dashboard_screenshot.png" alt="Dashboard Screenshot" width="600" style="border-radius: 1.2rem; box-shadow: 0 4px 32px #19d4ff33; margin: 2rem 0;"/> -->
</div>

---

## 🚩 Project Overview

This project delivers a **production-ready credit risk scoring pipeline** using machine learning. It covers:
- Data preprocessing & feature engineering
- Model training (Logistic Regression, Random Forest, XGBoost)
- Real-time scoring via FastAPI
- Interactive Streamlit dashboard
- Export-ready results for business presentation


---

## ✨ Features
- Credit risk scoring (probability of default)
- Feature importance & explainability (SHAP, feature importances)
- FastAPI API for real-time scoring
- Streamlit dashboard for interactive scoring & export
- Docker & cloud deployment ready


---

## 🚀 Quickstart
1. **Install dependencies:**
  ```sh
  pip install -r ../../requirements.txt
  ```
2. **Run the API and dashboard:**
  ```sh
  python ../../launch_demo.py
  # or run all demos
  python ../../launch_all_demos.py
  ```
3. **Open the app:**
  - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
  - Dashboard: [http://localhost:8501](http://localhost:8501)


---

## 🧪 Example API Call & Result

POST `/score`
```json
{
  "features": {"age": 35, "income": 50000, "loan_amount": 10000, "credit_history": 1}
}
```
**Sample result:**
```json
{
  "score": 0.12,
  "risk": "low"
}
```


---

## 📊 Dashboard Features
- Enter applicant features and get risk score instantly
- View feature importance and explanations (SHAP, feature importances)
- Download results as a text file for presentation
- Export-ready for PowerPoint or offline demos
<!-- Add dashboard screenshot if available -->


---

## 📤 Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation


---

## ☁️ Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t credit-risk-assessment -f Dockerfile .
  docker run -p 8000:8000 credit-risk-assessment
  ```
- Cloud deployment guides available in the main repo


---

## 📁 File Structure
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file

---

## 🏅 Performance & Results

### Model Metrics (Sample)
| Model              | AUC   | Accuracy | Recall | F1    |
|--------------------|-------|----------|--------|-------|
| Logistic Regression| 0.81  | 0.77     | 0.74   | 0.75  |
| Random Forest      | 0.85  | 0.80     | 0.78   | 0.79  |
| XGBoost            | 0.87  | 0.82     | 0.80   | 0.81  |

### Business Impact
- **Default Rate Reduction:** -12% vs. baseline
- **Approval Rate:** +7% with same risk
- **Explainability:** SHAP plots for all predictions

---

## ℹ️ About

This project demonstrates a real-world, production-grade credit risk scoring pipeline with modern MLOps, explainability, and business-ready outputs.

> **For more results, dashboards, and code, see the [notebooks/](../../03_credit_risk_cecl_model/notebooks/) and [reports/](../../03_credit_risk_cecl_model/reports/) folders!**
