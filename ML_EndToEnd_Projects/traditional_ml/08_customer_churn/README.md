# Customer Churn Prediction ML End-to-End

## Project Overview

This project provides a full pipeline for customer churn prediction using machine learning. It includes data preprocessing, feature engineering, model training (logistic regression, random forest, XGBoost), and exposes both API and dashboard for real-time scoring and export.

## Features
- Churn probability prediction
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
  "features": {"age": 42, "tenure": 24, "monthly_charges": 70, "contract": "month-to-month"}
}
```
**Sample result:**
```json
{
  "churn_probability": 0.27,
  "will_churn": false
}
```

## Dashboard Features
- Enter customer features and get churn probability instantly
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
  docker build -t customer-churn -f Dockerfile .
  docker run -p 8000:8000 customer-churn
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file

## Project Overview

This project provides a full pipeline for customer churn prediction using machine learning. It includes data preprocessing, feature engineering, model training (logistic regression, random forest, XGBoost), and exposes both API and dashboard for real-time scoring and export.

## Features
- Churn probability prediction
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
  "features": {"age": 42, "tenure": 24, "monthly_charges": 70, "contract": "month-to-month"}
}
```
**Sample result:**
```json
{
  "churn_probability": 0.27,
  "will_churn": false
}
```

## Dashboard Features
- Enter customer features and get churn probability instantly
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
  docker build -t customer-churn -f Dockerfile .
  docker run -p 8000:8000 customer-churn
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
