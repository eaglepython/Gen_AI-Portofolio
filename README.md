<div align="center">

# Joseph Bidias — Machine Learning and Generative AI

<img width="964" height="488" alt="Joseph Bidias — Machine Learning and Generative AI" src="https://github.com/user-attachments/assets/12fb6ea5-077e-433a-bf0e-d2afb772fd03" />

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=30&duration=2800&pause=2000&color=19D4FF&center=true&vCenter=true&width=940&lines=Machine+Learning+and+MLOps;Generative+AI+Engineering;End-to-End+ML+Projects;AI+Governance" alt="Typing SVG" />

[![Live Portfolio](https://img.shields.io/badge/Live_Portfolio-Experience-6c63ff?style=for-the-badge&logoColor=white)](https://eaglepython.github.io/Gen_AI-Portofolio/)
[![The 7th Sense Lab](https://img.shields.io/badge/The_7th_Sense_Lab-Hub-5DA9E9?style=for-the-badge)](https://the7thsenselab.vercel.app/)
[![Software Portfolio](https://img.shields.io/badge/Software_Portfolio-Full_Stack-00d4ff?style=for-the-badge)](https://eaglepython.github.io/Software-Engineer-Portofolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/joseph-bidias-eaglepython)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rodabeck777@gmail.com)

**Machine Learning and MLOps | Generative AI | AI Governance**

An end-to-end suite of sixteen runnable ML and GenAI projects, built on open models and standard datasets.

</div>

---

## About

I build machine learning and generative AI systems, from data and modeling through deployment and monitoring. This repository collects sixteen end-to-end projects, each its own runnable application built around real models and standard datasets. They are reference implementations meant to be read and run, not production deployments. The emphasis is on clean structure, honest documentation, and code someone else can pick up.

**Core areas**

- End-to-end ML pipelines, from data to a deployed service
- Generative AI with open transformer models
- MLOps with Docker, FastAPI, CI, and cloud deployment
- AI governance and responsible AI

---

## The project suite

Each project is a standalone Streamlit or FastAPI application. They run on sample or standard datasets and open model weights, so you can clone, install, and launch any one of them locally.

### Generative AI

| Project | What it is | Stack | Code |
|---|---|---|---|
| **AI Code Generator** | Natural language to code | Transformers, LangChain, FastAPI | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/gen_ai/11_ai_code_generator) |
| **AI Content Creator** | Text and image generation | GPT-style, T5, Stable Diffusion | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/gen_ai/12_ai_content_creator) |
| **Document Intelligence** | OCR and NLP document analysis | Tesseract, Transformers | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/gen_ai/13_document_intelligence) |
| **Conversational AI** | Chatbot with intent recognition, entity extraction, and dialogue | BERT NER, BlenderBot, FastAPI | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/gen_ai/14_conversational_ai) |
| **Drug Discovery AI** | Molecular property prediction and screening | PyTorch Geometric, RDKit | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/gen_ai/15_drug_discovery_ai) |

### Traditional ML

| Project | What it is | Stack | Code |
|---|---|---|---|
| **E-commerce Recommender** | Hybrid recommendation engine | TensorFlow, scikit-learn | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/01_ecommerce_recommender) |
| **Credit Risk Assessment** | Risk scoring with explainability | XGBoost, SHAP, FastAPI | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/02_credit_risk_assessment) |
| **Credit Risk CECL Model** | Expected credit loss modeling | Python, scikit-learn | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/03_credit_risk_cecl_model) |
| **Stock Forecasting** | Time-series prediction | LSTM, Prophet | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/04_stock_forecasting) |
| **Computer Vision** | Image classification with explanations | CNN, ViT, Grad-CAM | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/05_computer_vision) |
| **NLP Text Analysis** | Multi-task NLP | BERT, spaCy, Transformers | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/06_nlp_text_analysis) |
| **Fraud Detection** | Transaction fraud scoring | scikit-learn, imbalanced-learn | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/07_fraud_detection) |
| **Customer Churn** | Churn prediction | scikit-learn, XGBoost | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/08_customer_churn) |
| **Supply Chain Optimization** | Supply-chain analytics | Python, optimization | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/09_supply_chain_optimization) |
| **Energy Prediction** | Energy demand forecasting | Time-series models | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/10_energy_prediction) |
| **Autonomous Vehicle** | Computer vision for AV tasks | CNN, OpenCV | [View](https://github.com/eaglepython/Gen_AI-Portofolio/tree/main/ML_EndToEnd_Projects/traditional_ml/11_autonomous_vehicle) |

---

## Quick start

```bash
# Install dependencies
pip install -r ML_EndToEnd_Projects/requirements.txt

# Launch all demos
python launch_all_demos.py
# Each project serves on its own local port (8501, 8502, ...)

# Or launch one project
cd ML_EndToEnd_Projects/gen_ai/11_ai_code_generator
python launch_demo.py
```

---

## Technology stack

### Machine learning and AI
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logoColor=black)

### MLOps and infrastructure
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

### Data and web
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)

---

## Experience

- **Bidias Capital Consulting**, founder (2022 to present). Build and deploy ML and AI systems for public-sector and enterprise clients.
- **Verizon**, senior ML engineering (2022 to 2025). Production ML and real-time data pipelines at scale.
- **Texas Health and Human Services**, data and ML lead (2020 to 2022). Decision support across public health programs under state compliance rules.
- **University of Florida**, research data engineer (2017 to 2020). Statistical and ML modeling on patient datasets.

---

## Education and certifications

- M.S. Financial Engineering, WorldQuant University (in progress)
- B.S. Biomedical Engineering, High Institute of Medical Technologies, Yaoundé, Cameroon
- AWS Machine Learning Specialty
- Microsoft Certified: Data Scientist Associate
- AI Solution Architect, ELVTR
- AI2030 Global Fellowship, AI governance

---

## Writing

- [Bridging Innovation and Accountability: AI Governance in 2025](https://medium.com/@rodabeck777/bridging-innovation-and-accountability-the-path-forward-for-ai-governance-in-2025-9059eed8e3d2)
- [Financial Contagion Meets AI: Systemic Risk in Algorithmic Decision-Making](https://medium.com/@rodabeck777/financial-contagion-meets-ai-systemic-risk-in-algorithmic-decision-making-41c2fa87d0a3)
- [The Implementation Crisis: Why AI Policies Fail in Practice](https://medium.com/@rodabeck777/the-implementation-crisis-why-great-ai-policies-fail-in-practice-f91b19b0f30e)
- [Beyond Bias Audits: Building Continuous AI Accountability Systems](https://medium.com/@rodabeck777/beyond-bias-audits-building-continuous-ai-accountability-systems-d9f577ac4dac)

---

## Contact

- Email: rodabeck777@gmail.com
- Portfolio hub: [the7thsenselab.vercel.app](https://the7thsenselab.vercel.app/)
- GitHub: [github.com/eaglepython](https://github.com/eaglepython)
- Location: Dallas–Fort Worth, Texas, remote available

---

<div align="center">

Released under the MIT License. Projects run on sample or standard datasets and open model weights; they are reference implementations, not production deployments.

</div>
