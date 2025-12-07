# Salary Prediction ML Project

This project predicts employee salaries using machine learning, MLflow experiment tracking, and FastAPI deployment. It includes a complete workflow from model training to API serving.

## 1. Project Overview
This system uses the `employee_salary_dataset.csv` dataset to build and deploy a salary prediction model.

## 2. Dataset Description
- Employee_ID (removed from training)
- Department (categorical)
- Position (categorical)
- Age (numeric)
- Experience (numeric)
- Salary (target)

## 3. Model Training
Three models were trained:
- Linear Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)

Metrics used:
- MSE
- R² Score

Best Model: **Linear Regression**

## 4. MLflow Tracking
MLflow logs:
- Parameters
- Metrics
- Artifacts
- Model versions

Run MLflow UI:
```
mlflow ui
```
Visit:
```
http://127.0.0.1:5000
```

## 5. Deployment (FastAPI)
The best model is saved as `models/best_model.pkl`.

Run API:
```
uvicorn api.main:app --reload
```

Swagger Docs:
```
http://127.0.0.1:8000/docs
```

## 6. API Test Example
```
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"department\": \"IT\", \"experience\": 5, \"age\": 30, \"position\": \"Manager\"}"
```

## 7. Report Summary
- Linear Regression achieved the best performance.
- MLflow shows clear comparison among all models.
- FastAPI deployment successfully serves prediction requests.




