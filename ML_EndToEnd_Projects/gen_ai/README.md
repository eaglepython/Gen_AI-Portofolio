<div align="center">
  <h1 style="font-size:2.3rem; font-weight:900; color:#19d4ff; letter-spacing:0.04em; margin-bottom:0.5rem;">
    🤖 GenAI Projects — Advanced Generative AI Portfolio
  </h1>
  <p style="max-width:900px; color:#9aa8b2;">
    A visual portfolio of production-ready generative AI micro-projects — each includes a FastAPI backend, optional Streamlit dashboard, Docker support, and cloud deployment notes. Explore demos, APIs, and highlights for quick evaluation.
  </p>
  <!-- Replace the image below with a project collage or GIF: /assets/genai-collage.png -->
  ![Portfolio preview](./assets/genai-collage-placeholder.png)
</div>

---

## 🔍 At-a-glance (this folder)
This folder collects focused GenAI projects demonstrating different applications of foundation models: code generation, content creation, document intelligence, conversational agents, and drug-discovery prototypes. Each subfolder is a standalone mini-app with README, run instructions, and deployment artifacts.

Quick highlights:
- 5 focused projects (APIs + dashboards)
- Streamlit dashboards where applicable for rapid demos
- Dockerfiles + cloud hints for each project
- Example requests and demo scripts included

---

## 🗂️ Project Summary (visual cards)

### 1) 11_ai_code_generator
- Description: Convert natural-language prompts into runnable code snippets and full functions using code-focused LLMs.
- Key features: prompt templates, multi-language snippets, runnable examples, code safety heuristics.
- Quick run:
  - Backend: python app.py (FastAPI)
  - Dashboard: python dashboard.py (Streamlit, if present)
- Demo URL: ./11_ai_code_generator/README.md
- Tech: Python, FastAPI, OpenAI/Code-LLM, Docker
- Status: Demo-ready

---

### 2) 12_ai_content_creator
- Description: Automated multi-modal content generation — articles, SEO snippets, and image generation for social posts.
- Key features: content templates, batch generation, image + caption pairing, metadata for SEO.
- Quick run:
  - Backend: python app.py
  - Dashboard: python dashboard.py
- Demo URL: ./12_ai_content_creator/README.md
- Tech: Python, FastAPI, image-generation model + text LLM, Docker
- Status: Demo-ready

---

### 3) 13_document_intelligence
- Description: Parse, extract and analyze structured and unstructured documents (PDFs, images). Built for search, QA, and entity extraction.
- Key features: OCR, layout-aware parsing, semantic search, question-answering over documents.
- Quick run:
  - Backend: python app.py
  - Dashboard: python dashboard.py (document upload + QA)
- Demo URL: ./13_document_intelligence/README.md
- Tech: Python, FastAPI, OCR libs, embeddings, vector DB hints
- Status: Production prototype

---

### 4) 14_conversational_ai
- Description: Conversational agent framework with memory, tool invocation, and multi-turn dialogue support.
- Key features: session memory, tool interface, fallback handlers, analytics stubs.
- Quick run:
  - Backend: python app.py
  - Dashboard: python dashboard.py (chat UI)
- Demo URL: ./14_conversational_ai/README.md
- Tech: Python, FastAPI, LLMs, websocket/REST chat UI, Docker
- Status: Demo-ready with memory switches

---

### 5) 15_drug_discovery_ai
- Description: Research-oriented prototype applying generative models to propose candidate compounds and summarize assays.
- Key features: molecule suggestion, property prediction stubs, experiment summarization, safe-mode filters.
- Quick run:
  - Backend: python app.py
  - Analysis notebooks: ./15_drug_discovery_ai/notebooks
- Demo URL: ./15_drug_discovery_ai/README.md
- Tech: Python, scientific libs, molecular toolkits (RDKit), ML models
- Status: Research prototype

---

## 📌 Quick actions
- Launch all demos locally (example one-liner):
  - From this gen_ai folder: bash ./launch_all_local.sh
  - Or run specific project: cd 11_ai_code_generator && python launch_demo.py
- Start a single project:
  1. cd <project_folder>
  2. python -m venv .venv && source .venv/bin/activate
  3. pip install -r requirements.txt
  4. python app.py  # FastAPI
  5. python dashboard.py  # Streamlit (if provided)

---

## 🧩 Common components & patterns
- FastAPI backends (app.py) as the canonical API entrypoint.
- Optional Streamlit dashboards (dashboard.py) for non-technical demos.
- Dockerfiles for containerized runs and cloud deployment.
- Reusable prompt templates and a central utils/prompting module (see each project's README).
- Example requests and Postman/HTTPie snippets included per project.

---

## 📸 Gallery (replace placeholders with screenshots or GIFs)
- Code generator preview: ./assets/11_code_gen.png
- Content creator output: ./assets/12_content_creator.png
- Document QA demo: ./assets/13_doc_qna.png
- Conversational UI: ./assets/14_chat_ui.png
- Molecule proposal snapshot: ./assets/15_drug_disc.png

---

## ✅ What to expect inside each project folder
- README.md — project-specific overview and examples
- app.py — FastAPI service
- dashboard.py — Streamlit demo (if available)
- Dockerfile — build and run container
- requirements.txt — dependencies
- assets/ — images, example outputs
- notebooks/ — exploratory analyses (where present)

---

## ⚙️ Deployment notes
- Each project includes a Dockerfile — build with:
  docker build -t genai-<project> .
  docker run -p 8000:8000 genai-<project>
- Cloud: containerize + push to registry, then deploy to your cloud provider (GKE, ECS, App Service) or use serverless containers.
- Secrets: Use environment variables or secret stores for API keys (do not hardcode keys in repos).

---

## 🧭 Roadmap & contribution
- Polish dashboards with real screenshots and demo videos.
- Add unit tests and CI pipeline per project.
- Provide a unified gateway that lists running services and health checks.
- Contributions: open PRs to add demo media, fix README examples, or improve Docker configs.

---

## 📫 Contact & credits
- Author: eaglepython
- Repo: Gen_AI-Portofolio — ML_EndToEnd_Projects/gen_ai
- For quick queries: open an issue in the repo with label `genai-portfolio`.

> Explore each subfolder for detailed usage, sample responses, and test data. Replace the placeholder images in ./assets with real screenshots to make this portfolio even more visual.
