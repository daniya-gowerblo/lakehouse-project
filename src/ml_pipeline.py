import mlflow
import mlflow.sklearn
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from src.config import GOLD_PATH
import time

def run_ml():
    print("  Loading ML features from Gold layer...")
    
    try:
        df = pl.scan_delta(str(GOLD_PATH / "ml_features")).collect()
    except Exception as e:
        print(f"  Error reading Gold table: {e}")
        return
    
    if df.is_empty():
        print("  No data for ML.")
        return
    
    print(f"  Loaded {len(df)} rows for training.")
    
    pdf = df.to_pandas()
    numeric_cols = ['Distance', 'hour', 'DepDelay']
    available_cols = [c for c in numeric_cols if c in pdf.columns]
    
    if not available_cols:
        print("  Warning: No numeric features found.")
        return
        
    X = pdf[available_cols].fillna(0)
    target_reg = "ArrDelay"
    target_clf = "is_delayed"
    
    if target_reg not in pdf.columns or target_clf not in pdf.columns:
        print(f"  Error: Target columns missing.")
        return
        
    y_reg = pdf[target_reg]
    y_clf = pdf[target_clf]
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Flight_Lab3")
    
    with mlflow.start_run(run_name="Regression_Model"):
        print("  Training Regression Model...")
        reg_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        reg_model.fit(X_train, y_train_reg)
        
        pred_reg = reg_model.predict(X_test)
        mse = mean_squared_error(y_test_reg, pred_reg)
        
        mlflow.log_param("reg_model_type", "RandomForestRegressor")
        mlflow.log_param("features_used", str(available_cols))
        mlflow.log_metric("reg_mse", mse)
        mlflow.sklearn.log_model(reg_model, "regression_model")
        print(f"  Regression MSE: {mse:.2f}")

    time.sleep(1) 
    
    with mlflow.start_run(run_name="Classification_Model"):
        print("  Training Classification Model...")
        clf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        clf_model.fit(X_train, y_train_clf)
        
        pred_clf = clf_model.predict(X_test)
        acc = accuracy_score(y_test_clf, pred_clf)
        
        mlflow.log_param("clf_model_type", "RandomForestClassifier")
        mlflow.log_metric("clf_accuracy", acc)
        mlflow.sklearn.log_model(clf_model, "classification_model")
        print(f"  Classification Accuracy: {acc:.2%}")
        
    print("  ML Pipeline completed successfully!")