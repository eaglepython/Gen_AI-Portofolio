import joblib
import os
import numpy as np

class PDModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, 'models', 'pd_model.joblib')
        self.model = joblib.load(model_path)

    def predict(self, X):
        preds = self.model.predict(X)
        # Ensure output is in [0, 1]
        return np.clip(preds, 0, 1)
