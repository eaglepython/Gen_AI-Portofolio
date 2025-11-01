# Credit Risk & CECL Modeling Project

This project provides a full end-to-end pipeline for credit risk modeling and CECL (Current Expected Credit Loss) calculation, including:
- Data ingestion and processing
- Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) models
- API endpoints for model scoring (FastAPI)
- Interactive dashboard (Streamlit)
- Regulatory documentation and validation
- Automated tests and monitoring/logging

## Project Structure

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

## Quickstart

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

## Running Tests

1. Make sure all dependencies are installed.
2. From the project root, run:
   ```bash
   python -m pytest tests/
   ```
3. All model and API tests should pass if the API server is running (API tests auto-start the server if needed).

## Monitoring & Logging

- API logs: `logs/api.log`
- Dashboard logs: `logs/dashboard.log`

## Sample Data
See `data/raw/sample_credit_data.csv` for example input.

## Regulatory Compliance
- See `reports/regulatory_checklist.md` for compliance checklist.
- See `reports/model_documentation.md` and `reports/validation_report.md` for documentation and validation.

## Authors & License
- Joseph Bidias
- MIT License
