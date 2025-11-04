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
    - [`11_ai_code_generator/`](ML_EndToEnd_Projects/gen_ai/11_ai_code_generator/) — Code generation from natural language
    - [`12_ai_content_creator/`](ML_EndToEnd_Projects/gen_ai/12_ai_content_creator/) — Text & image content creation
    - [`13_document_intelligence/`](ML_EndToEnd_Projects/gen_ai/13_document_intelligence/) — Document analysis
    - [`14_conversational_ai/`](ML_EndToEnd_Projects/gen_ai/14_conversational_ai/) — Conversational AI/chatbots
    - [`15_drug_discovery_ai/`](ML_EndToEnd_Projects/gen_ai/15_drug_discovery_ai/) — Drug discovery AI
  - `traditional_ml/` — Classic ML projects
    - [`01_ecommerce_recommender/`](ML_EndToEnd_Projects/traditional_ml/01_ecommerce_recommender/) — Product recommendations
    - [`02_credit_risk_assessment/`](ML_EndToEnd_Projects/traditional_ml/02_credit_risk_assessment/) — Credit risk scoring
    - [`03_credit_risk_cecl_model/`](ML_EndToEnd_Projects/traditional_ml/03_credit_risk_cecl_model/) — Credit risk & CECL modeling
    - [`03_stock_forecasting/`](ML_EndToEnd_Projects/traditional_ml/03_stock_forecasting/) — Stock price prediction (legacy)
    - [`04_stock_forecasting/`](ML_EndToEnd_Projects/traditional_ml/04_stock_forecasting/) — Stock price prediction (current)
    - [`05_computer_vision/`](ML_EndToEnd_Projects/traditional_ml/05_computer_vision/) — Image classification and detection
    - [`06_nlp_text_analysis/`](ML_EndToEnd_Projects/traditional_ml/06_nlp_text_analysis/) — Sentiment, NER, summarization
    - [`07_fraud_detection/`](ML_EndToEnd_Projects/traditional_ml/07_fraud_detection/) — Fraud detection in transactions
    - [`08_customer_churn/`](ML_EndToEnd_Projects/traditional_ml/08_customer_churn/) — Churn prediction
    - [`09_supply_chain_optimization/`](ML_EndToEnd_Projects/traditional_ml/09_supply_chain_optimization/) — Supply chain analytics
    - [`10_energy_prediction/`](ML_EndToEnd_Projects/traditional_ml/10_energy_prediction/) — Energy consumption forecasting
    - [`11_autonomous_vehicle/`](ML_EndToEnd_Projects/traditional_ml/11_autonomous_vehicle/) — Autonomous vehicle ML
    - [`credit_risk_cecl_model/`](ML_EndToEnd_Projects/traditional_ml/credit_risk_cecl_model/) — Credit risk & CECL modeling suite
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

[![Portfolio Website](https://img.shields.io/badge/🌐_Portfolio_Website-LIVE-00d4ff?style=for-the-badge&logo=vercel&logoColor=white)](https://eaglepython.github.io/Gen_AI-Portofolio/)
[![The 7th Sense](https://img.shields.io/badge/🔮_The_7th_Sense-Interactive_Experience-6c63ff?style=for-the-badge)](https://eaglepython.github.io/Gen_AI-Portofolio/)

> **🚀 [LIVE PORTFOLIO WEBSITE](https://eaglepython.github.io/Gen_AI-Portofolio/) 🚀**
> 
> Experience the complete interactive portfolio with earthquake launch animation, spinning 3D profile, and all 16 ML/GenAI projects in a stunning visual presentation.

---

# 🏆 The 7th Sense: Complete ML/GenAI Portfolio

**Quantitative Researcher & Software Engineer** specializing in **AI/ML Solutions**, **Algorithmic Trading**, and **Full-Stack Development**

## 🌟 Interactive Portfolio Website

<div align="center">

### 🎯 **[VISIT LIVE WEBSITE](https://eaglepython.github.io/Gen_AI-Portofolio/)**

*Experience the earthquake launch animation, 3D effects, and interactive project showcases*

</div>

---

## 📊 Portfolio Analytics

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
   cd ML_EndToEnd_Projects/portfolio_website  cd ML_EndToEnd_Projects/portfolio_website
  npm run build
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
