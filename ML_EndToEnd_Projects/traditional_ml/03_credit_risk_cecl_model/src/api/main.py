from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from src.models.pd_model import PDModel
from src.models.lgd_model import LGDModel
from src.models.ead_model import EADModel
import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'api.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

app = FastAPI()

# Load models (replace with actual paths after training)
pd_model = PDModel()
lgd_model = LGDModel()
ead_model = EADModel()
# pd_model.load('models/pd_model.joblib')
# lgd_model.load('models/lgd_model.pkl')
# ead_model.load('models/ead_model.joblib')

class CreditInput(BaseModel):
    features: list

def log_request_response(request: Request, response: dict, endpoint: str):
    logging.info(f"Endpoint: {endpoint} | Request: {request} | Response: {response}")

@app.post("/score/pd")
async def score_pd(input: CreditInput, request: Request):
    X = pd.DataFrame([input.features])
    score = pd_model.predict(X)
    result = {"pd": float(score[0])}
    log_request_response(await request.body(), result, "/score/pd")
    return result

@app.post("/score/lgd")
async def score_lgd(input: CreditInput, request: Request):
    X = pd.DataFrame([input.features])
    score = lgd_model.predict(X)
    result = {"lgd": float(score[0])}
    log_request_response(await request.body(), result, "/score/lgd")
    return result

@app.post("/score/ead")
async def score_ead(input: CreditInput, request: Request):
    X = pd.DataFrame([input.features])
    score = ead_model.predict(X)
    result = {"ead": float(score[0])}
    log_request_response(await request.body(), result, "/score/ead")
    return result
