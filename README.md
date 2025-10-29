# ML & GenAI End-to-End Portfolio

This repository contains 15+ complete Machine Learning and Generative AI projects, each with:
- FastAPI backend (REST API)
- Streamlit dashboard for interactive demos
- Docker support and cloud deployment instructions
- One-click launch scripts
- Exportable results and charts for presentations

## Quick Start (All Projects)

1. **Install dependencies:**
   ```sh
   pip install -r ML_EndToEnd_Projects/requirements.txt
   ```
2. **One-click launch (all demos):**
   ```sh
   python launch_all_demos.py
   ```
   - This will start all backends and dashboards. Visit http://localhost:8501, 8502, ...
3. **Individual project launch:**
   ```sh
   python launch_demo.py
   ```
   (in the project folder)

---

## Project Structure

- `ML_EndToEnd_Projects/`
  - `gen_ai/` — Generative AI projects
    - `11_ai_code_generator/` — Code generation from natural language
    - `12_ai_content_creator/` — Text & image content creation
    - `13_document_intelligence/` — Document analysis
    - ...
  - `traditional_ml/` — Classic ML projects
    - `01_ecommerce_recommender/` — Product recommendations
    - `03_stock_forecasting/` — Stock price prediction
    - `05_nlp_text_analysis/` — Sentiment, NER, summarization
    - ...
  - `dashboard.py` — Central Streamlit landing page
  - `requirements.txt` — All dependencies
  - `Dockerfile` — Base Docker image for all projects

---

cdcdcd ML_EndToEnd_Projects/portfolio_website
npm run dev

### GenAI Projects

#### 11_ai_code_generator
- **Description:** Generate code from prompts using transformer models.
- **How to run:**
  - Backend: FastAPI (`app.py`)
  - Dashboard: Streamlit (`dashboard.py`)
  - Launch: `python launch_demo.py`
- **Features:**
  - Code generation for multiple languages
  - Export generated code
  - Docker & cloud ready

#### 12_ai_content_creator
- **Description:** Generate text and images using GPT, T5, Stable Diffusion, etc.
- **How to run:**
  - Backend: FastAPI (`app.py`)
  - Dashboard: Streamlit (`dashboard.py`)
  - Launch: `python launch_demo.py`
- **Features:**
  - Text/image generation
  - Sentiment, word count, charts
  - Export text, images, charts
  - Docker & cloud ready

#### 13_document_intelligence
- **Description:** Document parsing, extraction, and analysis.
- **How to run:**
  - Backend: FastAPI (`app.py`)
  - Launch: `python app.py`
- **Features:**
  - Document upload & analysis
  - API endpoints
  - Docker & cloud ready

... (repeat for other GenAI projects)

### Traditional ML Projects

#### 01_ecommerce_recommender
- **Description:** Personalized product recommendations.
- **How to run:**
  - Backend: FastAPI (`src/api/app.py`)
  - Dashboard: Streamlit (`dashboard.py`)
  - Launch: `python launch_demo.py`
- **Features:**
  - User-based recommendations
  - Export recommendations
  - Docker & cloud ready

#### 03_stock_forecasting
- **Description:** Stock price prediction and visualization.
- **How to run:**
  - Backend: FastAPI (`app.py`)
  - Dashboard: Streamlit (`dashboard.py`)
  - Launch: `python launch_demo.py`
- **Features:**
  - Time series forecasting
  - Downloadable forecast & charts
  - Docker & cloud ready

#### 05_nlp_text_analysis
- **Description:** Sentiment, NER, and summarization for text.
- **How to run:**
  - Backend: FastAPI (`app.py`)
  - Dashboard: Streamlit (`dashboard.py`)
  - Launch: `python launch_demo.py`
- **Features:**
  - Multiple NLP tasks
  - Export results
  - Docker & cloud ready

... (repeat for other ML projects)

---

## Docker & Cloud Deployment

- Each project includes a `Dockerfile` or uses the base Dockerfile.
- Build and run (example):
  ```sh
  docker build -t ai-content-creator -f ML_EndToEnd_Projects/gen_ai/12_ai_content_creator/Dockerfile .
  docker run -p 8000:8000 ai-content-creator
  ```
- Cloud deployment guides for Azure, AWS, GCP included in each project folder.

---

## Export & Presentation

- All dashboards allow one-click export of results, charts, and generated content.
- Use for live demos or save outputs for PowerPoint/portfolio.

---

## More Info

- See each project folder for detailed README, API docs, and usage examples.
- Central dashboard: `ML_EndToEnd_Projects/dashboard.py` (Streamlit)
- For questions or issues, see the main README or contact the maintainer.

---

## Portfolio Website & Resume

A modern interactive portfolio website is included:
- **Tech:** Next.js 14+, React, Framer Motion, Tailwind CSS
- **Features:**
  - Interactive project showcase with launch buttons for demos and APIs
  - Integrated professional resume (Joseph Bidias)
  - Links to all code, dashboards, and documentation
  - Responsive, dark-themed UI

**How to run:**
1. Navigate to the portfolio site folder:
   ```sh
   cd ML_EndToEnd_Projects/portfolio_website
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.

**Resume Section:**
- View the full professional resume and key project highlights directly on the site.
- All focus projects in this repo are linked and described in the Resume section.
