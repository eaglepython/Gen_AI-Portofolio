# Credit Risk & CECL Modeling Suite

A production-grade credit risk modeling suite for Commercial & Institutional Banking (CIB) portfolios, supporting Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) models. Includes CECL-compliant expected loss modeling, model validation, benchmarking, regulatory documentation, and operational integration.

## Features
- PD, LGD, EAD model development (logistic regression, survival analysis, XGBoost)
- CECL expected loss calculation with scenario analysis
- Model validation, benchmarking, and governance
- Regulatory documentation (SR 11-7, OCC 2011-12, Basel III, CECL)
- Portfolio analytics and risk reporting
- REST API and dashboard for business/risk teams
- Data pipelines and operational integration

## Technologies
Python, scikit-learn, xgboost, lifelines, pandas, FastAPI, Streamlit, SQL, Docker, Git, DVC (optional), MLflow, Jupyter, AWS (optional)

## Structure
- `data/` - Raw, processed, and external data
- `notebooks/` - EDA, modeling, validation, reporting
- `src/` - Modular code for data, features, models, validation, API, utils
- `reports/` - Model documentation, validation, regulatory reports

## Getting Started
1. Place raw data in `data/raw/`
2. Run EDA and modeling notebooks in `notebooks/`
3. Use `src/` modules for pipeline automation
4. Generate reports in `reports/`

---
