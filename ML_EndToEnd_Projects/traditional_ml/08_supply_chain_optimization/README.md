# Supply Chain Optimization ML End-to-End

## Project Overview

This project provides a full pipeline for supply chain optimization using machine learning and operations research. It includes demand forecasting, inventory optimization, route planning, and exposes both API and dashboard for real-time analysis and export.

## Features
- Demand forecasting (time series, ML)
- Inventory optimization (EOQ, safety stock)
- Route planning (VRP, TSP)
- FastAPI API for supply chain analytics
- Streamlit dashboard for interactive analysis and export
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

POST `/forecast`
```json
{
  "product_id": "A123",
  "days": 30
}
```
**Sample result:**
```json
{
  "forecast": [100, 110, 120, ...]
}
```

## Dashboard Features
- Enter product and time horizon, get demand forecast instantly
- Optimize inventory and routes interactively
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t supply-chain-optimization -f Dockerfile .
  docker run -p 8000:8000 supply-chain-optimization
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
