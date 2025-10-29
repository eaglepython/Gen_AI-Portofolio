
# Conversational AI

A chatbot system with intent recognition, entity extraction, dialogue management, and response generation using transformers and reinforcement learning.

## Features
- Intent recognition and entity extraction
- Dialogue management and context tracking
- Response generation with BlenderBot
- FastAPI API for chat integration
- Streamlit dashboard for interactive chat and export
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

POST `/chat`
```json
{
  "user_id": "user1",
  "message": "What's the weather today?"
}
```
**Sample result:**
```json
{
  "response": "The weather today is sunny with a high of 25°C."
}
```

## Dashboard Features
- Enter user ID and chat message
- View chatbot responses instantly
- Download chat history as a text file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download chat history for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t conversational-ai -f Dockerfile .
  docker run -p 8000:8000 conversational-ai
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard (if available)
- `README.md`: This file
