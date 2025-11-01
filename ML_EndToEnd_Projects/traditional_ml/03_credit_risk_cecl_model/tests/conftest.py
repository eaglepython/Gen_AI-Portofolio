import subprocess
import time
import pytest
import requests

API_URL = "http://127.0.0.1:8000/score/pd"

@pytest.fixture(scope="session", autouse=True)
def fastapi_server():
    # Start the server
    proc = subprocess.Popen([
        "python", "-m", "uvicorn", "src.api.main:app"
    ])
    # Wait for the server to be up
    for _ in range(30):
        try:
            requests.post(API_URL, json={"features": [0.5, 100000, 5, 1, 0]})
            break
        except Exception:
            time.sleep(1)
    yield
    proc.terminate()
    proc.wait()
