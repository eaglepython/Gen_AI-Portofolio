
# Fraud Detection ML End-to-End

## Project Overview

This project provides a real-time fraud detection pipeline for financial transactions. It includes anomaly detection, imbalanced data handling, streaming data processing, and exposes both API and dashboard for real-time scoring and export.

## Features
- Real-time fraud scoring (<50ms latency)
- Anomaly detection (Isolation Forest, Autoencoder, XGBoost)
- Imbalanced learning (SMOTE, ADASYN)
- Streaming data support (Kafka)
- Explainable decisions (SHAP)
- FastAPI API for fraud scoring
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
	"features": {"amount": 100, "country": "US", "device": "mobile", ...}
}
```
**Sample result:**
```json
{
	"fraud_probability": 0.92,
	"is_fraud": true,
	"explanation": {"amount": 0.7, "device": 0.2}
}
```

## Dashboard Features
- Enter transaction features and get fraud score instantly
- View SHAP explanations for each prediction
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
	```sh
	docker build -t fraud-detection -f Dockerfile .
	docker run -p 8000:8000 fraud-detection
	```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file