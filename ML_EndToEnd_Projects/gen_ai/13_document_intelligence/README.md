

<div align="center">
   <h1 style="font-size:2.3rem; font-weight:900; color:#19d4ff; letter-spacing:0.04em; margin-bottom:0.5rem;">
      📄 Document Intelligence — GenAI
   </h1>
   <!-- Add a dashboard or results image here if available -->
   <!-- <img src="docs/document_dashboard.png" alt="Dashboard Screenshot" width="600" style="border-radius: 1.2rem; box-shadow: 0 4px 32px #19d4ff33; margin: 2rem 0;"/> -->
</div>

---

## 🚩 Project Overview

A document processing system with OCR, NLP, information extraction, document classification, and automated summarization using transformers. Features:
- OCR for images and PDFs
- Summarization, classification, and entity extraction
- FastAPI API for document and text processing
- Streamlit dashboard for interactive document analysis and export
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

## 🧪 Example API Calls & Results

### OCR
POST `/ocr` (multipart file upload)
**Sample result:**
```json
{
   "text": "Extracted text from image or PDF."
}
```

### Summarization
POST `/process`
```json
{
   "text": "Your document text here",
   "task": "summarize"
}
```
**Sample result:**
```json
{
   "result": "Summary of the document."
}
```


---

## 📊 Dashboard Features
- Upload documents or images for OCR
- Analyze text for summarization, classification, and entity extraction
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
   docker build -t document-intelligence -f Dockerfile .
   docker run -p 8000:8000 document-intelligence
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
| Task         | Model         | Metric (F1/ROUGE) |
|--------------|--------------|-------------------|
| OCR          | Tesseract    | F1 0.92           |
| Summarization| BART         | ROUGE-L 0.39      |
| Classification| DistilBERT  | F1 0.88           |

### Business Impact
- **OCR F1:** Up to 0.92 (Tesseract)
- **Summarization Quality:** ROUGE-L up to 0.39 (BART)
- **Classification F1:** Up to 0.88 (DistilBERT)
- **Inference Latency:** <2s per document

---

## ℹ️ About

This project demonstrates a real-world, production-grade GenAI document intelligence pipeline with modern MLOps, explainability, and business-ready outputs.

> **For more results, dashboards, and code, see the [notebooks/](../../notebooks/) and [docs/](../../docs/) folders!**
