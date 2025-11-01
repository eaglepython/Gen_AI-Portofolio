import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, 'data', 'raw', 'sample_credit_data.csv')
model_path = os.path.join(BASE_DIR, 'models', 'ead_model.joblib')

data = pd.read_csv(data_path)
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
y = data['ead']

model = GradientBoostingRegressor()
model.fit(X, y)

joblib.dump(model, model_path)
print(f'EAD model trained and saved to {model_path}')
