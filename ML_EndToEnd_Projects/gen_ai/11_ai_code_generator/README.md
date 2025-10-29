
# AI Code Generator

A production-ready code generation system using transformers, CodeBERT, and GPT models for automated programming assistance, code completion, and documentation generation.

## Features
- Natural language to code generation
- Supports Python and other languages
- Multiple transformer models (CodeGen, CodeBERT, GPT-Neo, StarCoder)
- FastAPI API for integration
- Streamlit dashboard for interactive code generation and export
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

POST `/generate`
```json
{
  "prompt": "function to reverse a string",
  "language": "python"
}
```
**Sample result:**
```json
{
  "generated_code": "def reverse_string(s):\n    return s[::-1]"
}
```

## Dashboard Features
- Enter prompt and select language/model
- View generated code instantly
- Download code as a file for presentation
- Example screenshot:

![Dashboard Screenshot](dashboard_screenshot.png)

## Export & Presentation
- Use the export button to download generated code for PowerPoint or offline demos
- All results are ready for direct presentation

## Deployment
- Dockerfile included. Build and run:
  ```sh
  docker build -t ai-code-generator -f Dockerfile .
  docker run -p 8000:8000 ai-code-generator
  ```
- Cloud deployment guides available in the main repo

## Files
- `app.py`: FastAPI backend
- `dashboard.py`: Streamlit dashboard
- `README.md`: This file
