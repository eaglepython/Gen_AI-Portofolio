# Energy Consumption Prediction ML End-to-End

## Project Overview

This project provides a full pipeline for energy consumption prediction using machine learning. It includes data preprocessing, feature engineering, model training (regression, time series), and exposes both API and dashboard for real-time forecasting and export.

## Features
- Energy demand forecasting (regression, time series)
- Feature importance and explainability
- FastAPI API for prediction
- Streamlit dashboard for interactive forecasting and export
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

POST `/predict`
```json
{
  "building_id": "B101",
  "days": 7
}
```
**Sample result:**
```json
{
  "forecast": [120, 130, 125, ...]
}
```

## Dashboard Features
- Enter building and time horizon, get energy forecast instantly
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
  docker build -t energy-prediction -f Dockerfile .
  docker run -p 8000:8000 energy-prediction
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
