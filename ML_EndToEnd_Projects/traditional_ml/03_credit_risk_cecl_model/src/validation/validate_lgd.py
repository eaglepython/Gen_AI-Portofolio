import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(BASE_DIR, 'data', 'raw', 'sample_credit_data.csv')
model_path = os.path.join(BASE_DIR, 'models', 'lgd_model.joblib')
report_path = os.path.join(BASE_DIR, 'reports', 'validation_report.md')

data = pd.read_csv(data_path)
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
y_true = data['lgd']

model = joblib.load(model_path)
y_pred = model.predict(X)

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

report = f"""
## LGD Model Validation

- Mean Squared Error: {mse:.6f}
- R^2 Score: {r2:.4f}

"""

with open(report_path, 'a') as f:
    f.write(report)

print("Validation complete. Results appended to reports/validation_report.md")
