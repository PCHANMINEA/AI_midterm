# mlflow_tracking.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

print('Libraries loaded')

#load dataset
df= pd.read_csv('employee_salary_dataset.csv')
df.head()

print(df.shape)
print(df.dtypes)

TARGET='Monthly_Salary'
X = df.drop(columns=[TARGET])
y = df[TARGET]


cat_cols = [c for c in X.columns if X[c].dtypes == 'object']
num_cols = [c for c in X.columns if X[c].dtypes != 'object']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output= False), cat_cols)
    ]
)

X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pre)
X_test_scaled = scaler.transform(X_test_pre)



mlflow.set_experiment('salary_regression_experiment')

#Models

models = [
    ("LinearRegression", LinearRegression(), X_train_scaled, X_test_scaled, {'fit_intercept': True}),
    ("SVR", SVR(kernel='rbf', C=1.0), X_train_scaled, X_test_scaled, {'kernel':'rbf','C':1.0}),
    ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42), X_train_pre, X_test_pre, {'n_estimators':100})
]

for name, model, Xtr, Xte, params in models: 
    with mlflow.start_run(run_name=name): 
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        for K, V in params.items():
            mlflow.log_param(K, V)
        mlflow.log_metric('MSE', float(mse))
        mlflow.log_metric('R2', float(r2))
        mlflow.sklearn.log_model(model, name='model')
        print(f"{name} logged: MSE={mse:.4f}, R2={r2:.4f}")
