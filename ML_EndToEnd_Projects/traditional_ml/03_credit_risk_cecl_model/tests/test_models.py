import os
import joblib
import pandas as pd
import numpy as np

def test_pd_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'pd_model.joblib')
    data_path = os.path.join(base_dir, 'data', 'raw', 'sample_credit_data.csv')
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all((preds >= 0) & (preds <= 1)), "PD predictions should be probabilities between 0 and 1"

def test_lgd_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'lgd_model.joblib')
    data_path = os.path.join(base_dir, 'data', 'raw', 'sample_credit_data.csv')
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all((preds >= 0) & (preds <= 1)), "LGD predictions should be between 0 and 1"

def test_ead_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'ead_model.joblib')
    data_path = os.path.join(base_dir, 'data', 'raw', 'sample_credit_data.csv')
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all(preds >= 0), "EAD predictions should be non-negative"
