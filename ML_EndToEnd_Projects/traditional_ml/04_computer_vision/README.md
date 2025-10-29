# Computer Vision ML End-to-End

## Project Overview

This project provides a full pipeline for computer vision tasks such as image classification, object detection, and image augmentation. It includes data preprocessing, model training (CNN, transfer learning), and exposes both API and dashboard for real-time inference and export.

## Features
- Image classification (CNN, ResNet, EfficientNet)
- Object detection (YOLO, SSD)
- Image augmentation and preprocessing
- FastAPI API for image analysis
- Streamlit dashboard for interactive inference and export
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
- Upload an image file (multipart)
**Sample result:**
```json
{
  "class": "cat",
  "confidence": 0.97
}
```

## Dashboard Features
- Upload images for classification or detection
- View predictions and confidence scores instantly
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t computer-vision -f Dockerfile .
  docker run -p 8000:8000 computer-vision
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
