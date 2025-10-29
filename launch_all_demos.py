# One-Click Demo Launcher for All Projects
# This script will start all FastAPI backends and Streamlit dashboards for demo purposes.

import subprocess
import sys
import os
import time

# List of (backend_cmd, dashboard_cmd, project_name)
PROJECTS = [
    # (Backend, Dashboard, Name)
    ([sys.executable, '-m', 'uvicorn', 'ML_EndToEnd_Projects.traditional_ml.01_ecommerce_recommender.src.api.app:app', '--reload'],
     [sys.executable, '-m', 'streamlit', 'run', 'ML_EndToEnd_Projects/traditional_ml/01_ecommerce_recommender/dashboard.py'],
     'E-commerce Recommender'),
    ([sys.executable, '-m', 'uvicorn', 'ML_EndToEnd_Projects.traditional_ml.03_stock_forecasting.app:app', '--reload'],
     [sys.executable, '-m', 'streamlit', 'run', 'ML_EndToEnd_Projects/traditional_ml/03_stock_forecasting/dashboard.py'],
     'Stock Forecasting'),
    ([sys.executable, '-m', 'uvicorn', 'ML_EndToEnd_Projects.traditional_ml.05_nlp_text_analysis.app:app', '--reload'],
     [sys.executable, '-m', 'streamlit', 'run', 'ML_EndToEnd_Projects/traditional_ml/05_nlp_text_analysis/dashboard.py'],
     'NLP Text Analysis'),
    ([sys.executable, '-m', 'uvicorn', 'ML_EndToEnd_Projects.gen_ai.11_ai_code_generator.app:app', '--reload'],
     [sys.executable, '-m', 'streamlit', 'run', 'ML_EndToEnd_Projects/gen_ai/11_ai_code_generator/dashboard.py'],
     'AI Code Generator'),
    ([sys.executable, '-m', 'uvicorn', 'ML_EndToEnd_Projects.gen_ai.12_ai_content_creator.app:app', '--reload'],
     [sys.executable, '-m', 'streamlit', 'run', 'ML_EndToEnd_Projects/gen_ai/12_ai_content_creator/dashboard.py'],
     'AI Content Creator'),
]

backend_procs = []
try:
    for backend_cmd, dashboard_cmd, name in PROJECTS:
        print(f'Starting {name} backend...')
        proc = subprocess.Popen(backend_cmd)
        backend_procs.append(proc)
        time.sleep(3)
        print(f'Starting {name} dashboard...')
        subprocess.Popen(dashboard_cmd)
        time.sleep(2)
    print('\nAll demos are launching!')
    print('Visit the dashboards at http://localhost:8501, 8502, ...')
    print('API docs for each backend at http://localhost:8000, 8001, ...')
    print('\nPress Ctrl+C to stop all.')
    for proc in backend_procs:
        proc.wait()
except KeyboardInterrupt:
    print('Shutting down all demos...')
    for proc in backend_procs:
        proc.terminate()
