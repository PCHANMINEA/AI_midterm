
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = 'best_salary_model_joblib'
PRE_PATH = 'preprocessor.joblib'
SCALER_PATH = 'scaler.joblib'

if not os.path.exists(MODEL_PATH):
    print('Warning: model not found . Run training first.')

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
preprocessor = joblib.load(PRE_PATH) if os.path.exists(PRE_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

@app.get('/')
def root():
    return {'status' : 'ok', 'model_loaded': bool(model)}
@app.post('/predict')
def predict(payload: dict):
    if model is None or preprocessor is None or scaler is None:
        raise HTTPException(status_code=500, detail='Model or preprocessor missing. Run Training')
        df = pd.DataFrame([payload])
        x_pre = preprocessor.transform(df)
        x_scaled = scaler.transform(x_pre)
        pred = model.predict(x_scaled)[0]
        return {'predicted_salary': float(pred)}