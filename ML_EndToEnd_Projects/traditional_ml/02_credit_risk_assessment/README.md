# Credit Risk Assessment ML End-to-End

## Project Overview

This project provides a full pipeline for credit risk assessment using machine learning. It includes data preprocessing, feature engineering, model training (logistic regression, random forest, XGBoost), and exposes both API and dashboard for real-time scoring and export.

## Features
- Credit risk scoring (probability of default)
- Feature importance and explainability
- FastAPI API for scoring
- Streamlit dashboard for interactive scoring and export
- Docker & cloud deployment ready

## Usage
1. **Install dependencies:**
   ```sh
   pip install -r ../../requirements.txt
   ```
2. **Run the API and dashboard (one-click):**
   ```sh
   python ../../launch_demo.py
   ```
   or run all demos:
   ```sh
   python ../../launch_all_demos.py
   ```
3. **Open the app:**
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Dashboard: [http://localhost:8501](http://localhost:8501)

## Example API Call & Result

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

## Dashboard Features
- Enter applicant features and get risk score instantly
- View feature importance and explanations
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t credit-risk-assessment -f Dockerfile .
  docker run -p 8000:8000 credit-risk-assessment
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
