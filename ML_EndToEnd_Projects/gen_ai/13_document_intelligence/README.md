
# Document Intelligence

A document processing system with OCR, NLP, information extraction, document classification, and automated summarization using transformers.

## Features
- OCR for images and PDFs
- Summarization, classification, and entity extraction
- FastAPI API for document and text processing
- Streamlit dashboard for interactive document analysis and export
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

## Dashboard Features
- Upload documents or images for OCR
- Analyze text for summarization, classification, and entity extraction
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
   ```sh
   docker build -t document-intelligence -f Dockerfile .
   docker run -p 8000:8000 document-intelligence
   ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
