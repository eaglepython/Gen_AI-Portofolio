

<div align="center">
	<h1 style="font-size:2.7rem; font-weight:900; color:#19d4ff; letter-spacing:0.04em; margin-bottom:0.5rem;">
		<span style="font-size:2.2rem;">🚀</span> <span style="background:linear-gradient(90deg,#19d4ff,#6c63ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">ML &amp; GenAI End-to-End Portfolio</span> <span style="font-size:2.2rem;">🤖</span>
	</h1>
	<a href="https://eaglepython.github.io/Gen_AI-Portofolio/" target="_blank">
		<img src="https://img.shields.io/badge/Live%20Portfolio-View%20Now-19d4ff?style=for-the-badge&logo=vercel" alt="Live Portfolio" />
	</a>
	<br/>
	<a href="https://eaglepython.github.io/Gen_AI-Portofolio/" target="_blank">
		<img src="ML_EndToEnd_Projects/portfolio_website/public/portfolio_screenshot.png" alt="Portfolio Screenshot" width="700" style="border-radius: 1.2rem; box-shadow: 0 4px 32px #19d4ff33; margin: 2rem 0;"/>
	</a>
</div>

# ML End-to-End Projects Collection

A comprehensive collection of 15 production-ready machine learning projects covering traditional ML and generative AI approaches. Each project follows MLOps best practices with complete end-to-end pipelines from data collection to deployment and monitoring.

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
		- `02_credit_risk_assessment/` — Credit scoring and risk analysis
		- `03_credit_risk_cecl_model/` — Regulatory-compliant credit risk & CECL modeling
		- `03_stock_forecasting/` — Stock price prediction
		- `05_nlp_text_analysis/` — Sentiment, NER, summarization
		- ...
	- `dashboard.py` — Central Streamlit landing page
	- `requirements.txt` — All dependencies
	- `Dockerfile` — Base Docker image for all projects

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
	 npm install
	 npm run dev
	 ```
2. Open [http://localhost:3000](http://localhost:3000) in your browser.

**Resume Section:**
- View the full professional resume and key project highlights directly on the site.
- All focus projects in this repo are linked and described in the Resume section.