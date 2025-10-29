# One-Click Demo Launcher for AI Content Creator
# This script will start both the FastAPI backend and the Streamlit dashboard for demo purposes.

import subprocess
import sys
import os
import time

BACKEND_CMD = [sys.executable, '-m', 'uvicorn', 'ML_EndToEnd_Projects.gen_ai.12_ai_content_creator.app:app', '--reload']
DASHBOARD_CMD = [sys.executable, '-m', 'streamlit', 'run', 'ML_EndToEnd_Projects/gen_ai/12_ai_content_creator/dashboard.py']

backend_proc = None
try:
    print('Starting FastAPI backend...')
    backend_proc = subprocess.Popen(BACKEND_CMD)
    time.sleep(5)  # Give backend time to start
    print('Starting Streamlit dashboard...')
    subprocess.Popen(DASHBOARD_CMD)
    print('\nDemo is launching!')
    print('API docs: http://localhost:8000/docs')
    print('Dashboard: http://localhost:8501')
    print('\nPress Ctrl+C to stop.')
    backend_proc.wait()
except KeyboardInterrupt:
    print('Shutting down...')
    if backend_proc:
        backend_proc.terminate()
