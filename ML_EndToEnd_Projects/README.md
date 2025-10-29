# ML End-to-End Projects Collection

A comprehensive collection of 15 production-ready machine learning projects covering traditional ML and generative AI approaches. Each project follows MLOps best practices with complete end-to-end pipelines from data collection to deployment and monitoring.

## 🏗️ Architecture Overview

All projects follow a standardized architecture pattern:
- **Modular Pipelines**: Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve
- **MLflow Tracking**: Experiment management and model versioning
- **Cloud Deployment**: AWS/Azure integration with containerized services
- **API Services**: FastAPI REST endpoints for model serving
- **Interactive Dashboards**: Streamlit applications for data exploration and predictions
- **Comprehensive Testing**: Unit, integration, and end-to-end test suites

## 📊 Traditional ML Projects (10)

### 1. [E-commerce Recommender System](./traditional_ml/01_ecommerce_recommender/)
Hybrid recommendation engine using collaborative filtering, content-based filtering, and deep learning approaches.
- **Algorithms**: Matrix Factorization, Neural Collaborative Filtering, Content-based filtering
- **Features**: Real-time recommendations, A/B testing framework, cold-start handling
- **Stack**: Python, scikit-learn, TensorFlow, FastAPI, Streamlit, MLflow

### 2. [Credit Risk Assessment](./traditional_ml/02_credit_risk/)
Binary classification for loan default prediction with interpretable machine learning.
- **Algorithms**: XGBoost, Random Forest, Logistic Regression
- **Features**: SHAP explanations, fairness analysis, regulatory compliance
- **Stack**: Python, XGBoost, SHAP, FastAPI, Streamlit, MLflow

### 3. [Time Series Forecasting - Stock Market](./traditional_ml/03_stock_forecasting/)
Multi-step ahead forecasting with multiple time series models and technical indicators.
- **Algorithms**: LSTM, ARIMA, Prophet, Transformer
- **Features**: Technical indicators, sentiment analysis, real-time predictions
- **Stack**: Python, PyTorch, Prophet, FastAPI, Streamlit, Alpha Vantage API

### 4. [Computer Vision - Medical Image Classification](./traditional_ml/04_medical_imaging/)
CNN-based classification of medical images with transfer learning and explainability.
- **Algorithms**: ResNet, EfficientNet, Vision Transformer
- **Features**: Data augmentation, Grad-CAM, DICOM support

# ML & GenAI End-to-End Portfolio

Welcome! This portfolio contains 15+ real-world Machine Learning and Generative AI projects, each designed for easy demo, job interviews, and daily reference. Every project is:
- **Easy to run**: One-click launch for both backend (API) and dashboard
- **Well-documented**: Clear instructions, sample results, and screenshots
- **Presentation-ready**: Exportable results and charts for slides or reports
- **Production-minded**: Docker/cloud deployment, modular code, and best practices

---

## 🚀 Quick Start (All Projects)

1. **Install dependencies:**
	 ```sh
	 pip install -r ML_EndToEnd_Projects/requirements.txt
	 ```
2. **One-click launch (all demos):**
	 ```sh
	 python launch_all_demos.py
	 ```
	 - Starts all backends and dashboards. Visit http://localhost:8501, 8502, ...
3. **Individual project launch:**
	 ```sh
	 python launch_demo.py
	 ```
	 (in the project folder)

---

## 📂 Project Structure & Highlights

### GenAI Projects (`gen_ai/`)

- **AI Code Generator**: Generate code from natural language prompts. Supports Python and more. [Details](gen_ai/11_ai_code_generator/README.md)

- **AI Content Creator**: Generate text and images (GPT, T5, Stable Diffusion). [Details](gen_ai/12_ai_content_creator/README.md)

- **Document Intelligence**: OCR, summarization, entity extraction for documents. [Details](gen_ai/13_document_intelligence/README.md)

- **Conversational AI**: Chatbot with intent/entity recognition and dialogue. [Details](gen_ai/14_conversational_ai/README.md)

- **Drug Discovery AI**: Molecular property prediction and optimization. [Details](gen_ai/15_drug_discovery_ai/README.md)

### Traditional ML Projects (`traditional_ml/`)

- **E-commerce Recommender**: Personalized product recommendations. [Details](traditional_ml/01_ecommerce_recommender/README.md)

- **Credit Risk Assessment**: Predict loan default risk. [Details](traditional_ml/02_credit_risk_assessment/README.md)

- **Stock Forecasting**: Predict stock prices and visualize trends. [Details](traditional_ml/03_stock_forecasting/README.md)

- **NLP Text Analysis**: Sentiment, NER, summarization. [Details](traditional_ml/05_nlp_text_analysis/README.md)

...and more: Computer Vision, Fraud Detection, Churn, Supply Chain, Energy, Autonomous Vehicles.

---

## �️ How to Use for Interviews & Daily Work

- **Live Demo**: Run any project, show results, and export outputs for your slides or reports.
- **Code Reference**: Each folder has modular, production-style code for real-world tasks.
- **API & Dashboard**: Every project exposes a REST API and a Streamlit dashboard for hands-on use.
- **Export**: All dashboards have one-click export for results, charts, and generated content.
- **Deployment**: Dockerfiles and cloud guides included for each project.

---

## 🐳 Docker & Cloud Deployment

- Each project includes a `Dockerfile` or uses the base Dockerfile.
- Build and run (example):
	```sh
	docker build -t ai-content-creator -f gen_ai/12_ai_content_creator/Dockerfile .
	docker run -p 8000:8000 ai-content-creator
	```
- Cloud deployment guides for Azure, AWS, GCP included in each project folder.

---

## 📊 Export & Presentation

- All dashboards allow one-click export of results, charts, and generated content.
- Use for live demos, job interviews, or to save outputs for PowerPoint/portfolio.

---

## � More Info & Support

- See each project folder for detailed README, API docs, and usage examples.
- Central dashboard: `dashboard.py` (Streamlit)
- For questions or issues, see the main README or contact the maintainer.