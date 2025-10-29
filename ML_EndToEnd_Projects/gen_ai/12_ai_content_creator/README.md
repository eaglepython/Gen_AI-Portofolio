
# AI Content Creator

Automated content generation for text and images using GPT, T5, Stable Diffusion, and more.

## Features
- Text generation (blog posts, social media, creative writing)
- Image generation (Stable Diffusion, SDXL)
- FastAPI API for both text and image content
- Streamlit dashboard with analytics and export
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

### Text Generation
POST `/generate/text`
```json
{
  "prompt": "Write a poem about AI"
}
```
**Sample result:**
```
Roses are circuits, violets are code,
AI writes poems in digital mode.
Learning and growing, it never will tire,
Creating new verses to always inspire.
```

### Image Generation
POST `/generate/image`
```json
{
  "prompt": "A futuristic cityscape"
}
```
**Sample result:**
Image(s) returned as base64-encoded PNG. View and download in the dashboard.

## Dashboard Features
- Generate text and images interactively
- View word count, sentiment, and word length charts for text
- Download generated text, images, and charts with one click
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export buttons to download results for PowerPoint or offline demos
- All results and visuals are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t ai-content-creator -f Dockerfile .
  docker run -p 8000:8000 ai-content-creator
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard
- `README.md`: This file