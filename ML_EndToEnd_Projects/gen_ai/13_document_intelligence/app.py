"""
Document Intelligence System - End-to-End Implementation
OCR, NLP, information extraction, classification, and summarization using transformers.
"""

import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import torch
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

# NLP models
SUMMARIZATION_MODEL = 'facebook/bart-large-cnn'
NER_MODEL = 'dbmdz/bert-large-cased-finetuned-conll03-english'
CLASSIFICATION_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'

summarizer = pipeline('summarization', model=SUMMARIZATION_MODEL, device=0 if torch.cuda.is_available() else -1)
ner_pipe = pipeline('ner', model=NER_MODEL, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
classifier = pipeline('text-classification', model=CLASSIFICATION_MODEL, device=0 if torch.cuda.is_available() else -1)

# FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="OCR, NLP, information extraction, classification, and summarization using transformers.",
    version="1.0.0"
)

# Request/response models
class TextRequest(BaseModel):
    text: str
    task: str = 'summarize'  # summarize, classify, extract

class TextResponse(BaseModel):
    result: Dict
    task: str

@app.post("/ocr")
async def ocr_document(file: UploadFile = File(...)):
    """Extract text from uploaded image or PDF using OCR."""
    if not pytesseract or not Image:
        raise HTTPException(status_code=500, detail="OCR dependencies not installed.")
    try:
        suffix = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            img = Image.open(tmp_path)
            text = pytesseract.image_to_string(img)
        elif suffix == '.pdf':
            from pdf2image import convert_from_path
            images = convert_from_path(tmp_path)
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        os.unlink(tmp_path)
        return {"text": text}
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=TextResponse)
async def process_text(request: TextRequest):
    """Process text for summarization, classification, or NER extraction."""
    try:
        if request.task == 'summarize':
            summary = summarizer(request.text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            return TextResponse(result={"summary": summary}, task='summarize')
        elif request.task == 'classify':
            label = classifier(request.text)[0]
            return TextResponse(result={"label": label}, task='classify')
        elif request.task == 'extract':
            entities = ner_pipe(request.text)
            return TextResponse(result={"entities": entities}, task='extract')
        else:
            raise HTTPException(status_code=400, detail="Invalid task.")
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Document Intelligence API", "docs": "/docs"}

# Example usage (CLI)
def main():
    print("\n=== Document Intelligence System ===")
    mode = input("Choose [ocr/text]: ").strip().lower()
    if mode == 'ocr':
        print("OCR demo: Please use the API to upload files.")
    elif mode == 'text':
        text = input("Enter your document text: ")
        task = input("Task [summarize/classify/extract]: ") or 'summarize'
        if task == 'summarize':
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            print("\nSummary:\n", summary)
        elif task == 'classify':
            label = classifier(text)[0]
            print("\nClassification:\n", label)
        elif task == 'extract':
            entities = ner_pipe(text)
            print("\nEntities:\n", entities)
        else:
            print("Invalid task.")
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()
