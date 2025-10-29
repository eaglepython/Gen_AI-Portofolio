# ML & GenAI Project Dashboard

Welcome to the ML & GenAI Project Suite! This dashboard provides a summary, usage instructions, and quick links for all 15 end-to-end projects.

---

## 🚀 Project List & Usage

### 1. E-commerce Recommender
- **Path:** `traditional_ml/01_ecommerce_recommender/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 2. Credit Risk Assessment
- **Path:** `traditional_ml/02_credit_risk_assessment/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 3. Stock Forecasting
- **Path:** `traditional_ml/03_stock_forecasting/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 4. Computer Vision System
- **Path:** `traditional_ml/04_computer_vision/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 5. NLP Text Analysis
- **Path:** `traditional_ml/05_nlp_text_analysis/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 6. Fraud Detection
- **Path:** `traditional_ml/06_fraud_detection/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 7. Customer Churn
- **Path:** `traditional_ml/07_customer_churn/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 8. Supply Chain Optimization
- **Path:** `traditional_ml/08_supply_chain_optimization/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 9. Energy Prediction
- **Path:** `traditional_ml/09_energy_prediction/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 10. Autonomous Vehicle System
- **Path:** `traditional_ml/10_autonomous_vehicle/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 11. AI Code Generator
- **Path:** `gen_ai/11_ai_code_generator/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 12. AI Content Creator
- **Path:** `gen_ai/12_ai_content_creator/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 13. Document Intelligence
- **Path:** `gen_ai/13_document_intelligence/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 14. Conversational AI
- **Path:** `gen_ai/14_conversational_ai/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

### 15. Drug Discovery AI
- **Path:** `gen_ai/15_drug_discovery_ai/app.py`
- **Run:** `uvicorn app:app --reload`
- **API Docs:** `/docs`

---

## 🛠️ How to Run Any Project

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run a project:**
   ```sh
   cd <project_folder>
   uvicorn app:app --reload
   ```
3. **Open browser:**
   - Visit `http://localhost:8000/docs` for interactive API docs

---

## 📦 Docker Usage

1. **Build image:**
   ```sh
   docker build -t ml-genai-suite .
   ```
2. **Run container:**
   ```sh
   docker run -p 8000:8000 ml-genai-suite
   ```

---

## 📊 Monitoring & CI/CD
- Integrate with tools like Prometheus, Grafana, or GitHub Actions for monitoring and CI/CD.
- Add `prometheus_fastapi_instrumentator` to requirements for metrics.
- Example GitHub Actions workflow can be provided on request.

---

## 📚 Documentation
- Each project contains a `README.md` with details, usage, and API endpoints.
- For more help, open the project folder and review the code or ask for specific instructions.

---

Enjoy your complete ML & GenAI project suite!
