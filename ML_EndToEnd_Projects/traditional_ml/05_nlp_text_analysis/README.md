# NLP Text Analysis ML End-to-End

## Project Overview

This project provides a complete pipeline for NLP text analysis, including sentiment analysis, named entity recognition (NER), and text summarization. It combines state-of-the-art transformer models with traditional NLP techniques, and exposes both API and interactive dashboard for real-time analysis and export.

## Features
- Sentiment analysis (positive/negative/neutral)
- Named entity recognition (NER)
- Text summarization
- FastAPI API for all tasks
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

## Example API Calls & Results

### Sentiment Analysis
POST `/analyze` with:
```json
{
  "text": "Streamlit is an awesome tool for ML demos!",
  "task": "sentiment"
}
```
**Sample result:**
```json
{
  "result": {"label": "positive", "score": 0.98}
}
```

### NER
POST `/analyze` with:
```json
{
  "text": "Barack Obama was the 44th President of the United States.",
  "task": "ner"
}
```
**Sample result:**
```json
{
  "result": [{"entity": "Barack Obama", "type": "PERSON"}, {"entity": "United States", "type": "LOCATION"}]
}
```

### Summarization
POST `/analyze` with:
```json
{
  "text": "Natural language processing (NLP) is a field of AI...",
  "task": "summarize"
}
```
**Sample result:**
```json
{
  "result": "NLP is a field of AI focused on language understanding."
}
```

## Dashboard Features
- Enter text and select task (sentiment, NER, summarize)
- View results instantly
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t nlp-text-analysis -f Dockerfile .
  docker run -p 8000:8000 nlp-text-analysis
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard
- `README.md`: This file
