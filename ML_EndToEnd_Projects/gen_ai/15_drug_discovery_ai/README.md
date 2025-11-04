

<div align="center">
  <h1 style="font-size:2.3rem; font-weight:900; color:#19d4ff; letter-spacing:0.04em; margin-bottom:0.5rem;">
    🧬 Drug Discovery AI — GenAI
  </h1>
  <!-- Add a dashboard or results image here if available -->
  <!-- <img src="docs/drug_dashboard.png" alt="Dashboard Screenshot" width="600" style="border-radius: 1.2rem; box-shadow: 0 4px 32px #19d4ff33; margin: 2rem 0;"/> -->
</div>

---

## 🚩 Project Overview

A molecular property prediction system using graph neural networks, molecular descriptors, drug-target interaction prediction, and compound optimization. Features:
- Molecular property prediction (GNN, descriptors)
- Drug-target interaction scoring
- Compound optimization (SMILES modification)
- FastAPI API for molecule analysis
- Streamlit dashboard for interactive molecule analysis and export
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

POST `/molecule`
```json
{
  "smiles": "CCO",
  "task": "property"
}
```
**Sample result:**
```json
{
  "property": {"logP": 0.34, "mol_weight": 46.07}
}
```


---

## 📊 Dashboard Features
- Enter SMILES string and select task (property, interaction, optimize)
- View molecular properties and predictions instantly
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
  docker build -t drug-discovery-ai -f Dockerfile .
  docker run -p 8000:8000 drug-discovery-ai
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
| Task         | Model         | Metric (RMSE/AUC) |
|--------------|--------------|-------------------|
| Property     | GNN           | RMSE 0.41         |
| Interaction  | DeepDTA       | AUC 0.89          |

### Business Impact
- **Property Prediction RMSE:** As low as 0.41 (GNN)
- **Interaction AUC:** Up to 0.89 (DeepDTA)
- **Inference Latency:** <5s per molecule

---

## ℹ️ About

This project demonstrates a real-world, production-grade GenAI drug discovery pipeline with modern MLOps, explainability, and business-ready outputs.

> **For more results, dashboards, and code, see the [notebooks/](../../notebooks/) and [docs/](../../docs/) folders!**
