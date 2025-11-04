
<div align="center">
   <h1 style="font-size:2.3rem; font-weight:900; color:#19d4ff; letter-spacing:0.04em; margin-bottom:0.5rem;">
      🏦 Credit Risk & CECL Modeling — End-to-End ML
   </h1>
   <!-- Add a dashboard or results image here if available -->
   <!-- <img src="docs/dashboard_screenshot.png" alt="Dashboard Screenshot" width="600" style="border-radius: 1.2rem; box-shadow: 0 4px 32px #19d4ff33; margin: 2rem 0;"/> -->
</div>

---

## 🚩 Project Overview

This project provides a **production-grade, regulatory-compliant pipeline** for credit risk modeling and CECL (Current Expected Credit Loss) calculation, including:
- Data ingestion and processing
- Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) models
- API endpoints for model scoring (FastAPI)
- Interactive dashboard (Streamlit)
- Regulatory documentation and validation
- Automated tests and monitoring/logging


---

## 📁 Project Structure

```
03_credit_risk_cecl_model/
├── data/
│   └── raw/
│       └── sample_credit_data.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_pd_model.ipynb
│   ├── 03_lgd_model.ipynb
│   ├── 04_ead_model.ipynb
│   ├── 05_cecl_calculation.ipynb
│   ├── 06_validation.ipynb
│   ├── 07_reporting.ipynb
│   └── 08_api_and_dashboard_demo.ipynb
├── reports/
│   ├── model_documentation.md
│   ├── validation_report.md
│   └── regulatory_checklist.md
├── requirements.txt
├── README.md
├── src/
│   ├── api/
│   │   └── main.py
│   ├── dashboard/
│   │   └── app.py
│   ├── models/
│   │   ├── pd_model.py
│   │   ├── lgd_model.py
│   │   └── ead_model.py
│   ├── training/
│   └── validation/
├── tests/
│   ├── test_models.py
│   └── test_api.py
├── logs/
│   ├── api.log
│   └── dashboard.log
```


---

## 🚀 Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the API**
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```
3. **Run the dashboard**
   ```bash
   streamlit run src/dashboard/app.py
   ```
4. **Try the API**
   - Use the notebook `notebooks/08_api_and_dashboard_demo.ipynb` for examples.


---

## 🧪 Running Tests

1. Make sure all dependencies are installed.
2. From the project root, run:
   ```bash
   python -m pytest tests/
   ```
3. All model and API tests should pass if the API server is running (API tests auto-start the server if needed).


---

## 📊 Monitoring & Logging

- API logs: `logs/api.log`
- Dashboard logs: `logs/dashboard.log`


---

## 🗃️ Sample Data
See `data/raw/sample_credit_data.csv` for example input.


---

## 🛡️ Regulatory Compliance
- See `reports/regulatory_checklist.md` for compliance checklist.
- See `reports/model_documentation.md` and `reports/validation_report.md` for documentation and validation.


---

## 🏅 Model Performance & Validation

### Validation Metrics
| Model | MSE | R² Score |
|-------|---------|---------|
| PD    | 0.0000  | 1.0000  |
| LGD   | 0.0004  | 0.9211  |
| EAD   | 1.1460  | 1.0000  |

### Example API Usage
```python
import requests
payload = {"features": [0.5, 100000, 5, 1, 0]}
pd_response = requests.post("http://localhost:8000/score/pd", json=payload)
lgd_response = requests.post("http://localhost:8000/score/lgd", json=payload)
ead_response = requests.post("http://localhost:8000/score/ead", json=payload)
print("PD Score:", pd_response.json())
print("LGD Score:", lgd_response.json())
print("EAD Score:", ead_response.json())
```

---

## 📊 Dashboard & API Demo
- Launch the dashboard: `streamlit run src/dashboard/app.py`
- Try the API: see `notebooks/08_api_and_dashboard_demo.ipynb`

---

## ℹ️ About

- Author: Joseph Bidias
- License: MIT

> **For more results, dashboards, and code, see the [notebooks/](notebooks/) and [reports/](reports/) folders!**
