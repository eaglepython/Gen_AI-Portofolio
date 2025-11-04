
# Autonomous Vehicle ML End-to-End

## Project Overview

This project provides a full pipeline for autonomous vehicle perception and control using machine learning and deep learning. It includes sensor data preprocessing, object detection, lane detection, path planning, and exposes both API and dashboard for real-time inference and export.

## Features
- Object detection (YOLO, SSD)
- Lane detection (CNN, classical CV)
- Path planning (A*, Dijkstra, RL)
- Sensor fusion (camera, lidar, radar)
- FastAPI API for perception and planning
- Streamlit dashboard for interactive demo and export
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

POST `/detect`
- Upload an image or sensor data (multipart)
**Sample result:**
```json
{
  "objects": [{"type": "car", "confidence": 0.98, "bbox": [100, 200, 150, 250]}],
  "lanes": [[10, 100], [20, 200], ...]
}
```

## Dashboard Features
- Upload images or sensor data for detection and planning
- View detected objects, lanes, and planned path instantly
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t autonomous-vehicle -f Dockerfile .
  docker run -p 8000:8000 autonomous-vehicle
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file

## Project Overview

This project provides a full pipeline for autonomous vehicle perception and control using machine learning and deep learning. It includes sensor data preprocessing, object detection, lane detection, path planning, and exposes both API and dashboard for real-time inference and export.

## Features
- Object detection (YOLO, SSD)
- Lane detection (CNN, classical CV)
- Path planning (A*, Dijkstra, RL)
- Sensor fusion (camera, lidar, radar)
- FastAPI API for perception and planning
- Streamlit dashboard for interactive demo and export
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

POST `/detect`
- Upload an image or sensor data (multipart)
**Sample result:**
```json
{
  "objects": [{"type": "car", "confidence": 0.98, "bbox": [100, 200, 150, 250]}],
  "lanes": [[10, 100], [20, 200], ...]
}
```

## Dashboard Features
- Upload images or sensor data for detection and planning
- View detected objects, lanes, and planned path instantly
- Download results as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download results for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t autonomous-vehicle -f Dockerfile .
  docker run -p 8000:8000 autonomous-vehicle
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file