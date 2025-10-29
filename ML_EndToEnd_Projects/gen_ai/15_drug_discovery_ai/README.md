
# Drug Discovery AI

A molecular property prediction system using graph neural networks, molecular descriptors, drug-target interaction prediction, and compound optimization.

## Features
- Molecular property prediction (GNN, descriptors)
- Drug-target interaction scoring
- Compound optimization (SMILES modification)
- FastAPI API for molecule analysis
- Streamlit dashboard for interactive molecule analysis and export
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

## Dashboard Features
- Enter SMILES string and select task (property, interaction, optimize)
- View molecular properties and predictions instantly
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t drug-discovery-ai -f Dockerfile .
  docker run -p 8000:8000 drug-discovery-ai
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
