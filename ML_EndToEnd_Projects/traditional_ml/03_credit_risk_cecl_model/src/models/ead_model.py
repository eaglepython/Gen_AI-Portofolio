import joblib
import os
import numpy as np

class EADModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, 'models', 'ead_model.joblib')
        self.model = joblib.load(model_path)

    def predict(self, X):
        preds = self.model.predict(X)
        # EAD should be non-negative
        return np.maximum(preds, 0)
