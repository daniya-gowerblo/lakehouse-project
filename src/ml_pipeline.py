import json
import os
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
import pandas as pd
from deltalake import DeltaTable
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.config import GOLD_PATH, ML_SAMPLE_ROWS


def _make_preprocessor(pdf):
    numeric_cols = [
        c for c in [
            "distance",
            "hour",
            "day_of_week",
            "origin_day_flights",
            "dest_day_flights",
            "airline_day_flights",
            "route_day_flights",
            "origin_hour_flights",
            "route_hour_flights",
        ]
        if c in pdf.columns
    ]
    categorical_cols = [
        c for c in ["airline", "origin", "dest", "season", "route"]
        if c in pdf.columns
    ]

    encoder = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols + categorical_cols


def _regression_metrics(y_true, prediction):
    mse = mean_squared_error(y_true, prediction)
    return {
        "mae": mean_absolute_error(y_true, prediction),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": r2_score(y_true, prediction),
    }


def _classification_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    return model.predict(X_test)


def _classification_metrics(y_true, prediction, score):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "confusion_tn": int(tn),
        "confusion_fp": int(fp),
        "confusion_fn": int(fn),
        "confusion_tp": int(tp),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, score)
    except ValueError:
        metrics["roc_auc"] = 0.5

    return metrics


def _log_feature_importance(model, run_prefix):
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]

    if not hasattr(estimator, "feature_importances_"):
        return

    feature_names = list(preprocessor.get_feature_names_out())
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, f"{run_prefix}_feature_importance.csv")
        importance.to_csv(path, index=False)
        mlflow.log_artifact(path)

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

    if len(df) > ML_SAMPLE_ROWS:
        print(f"  Sampling {ML_SAMPLE_ROWS} rows for ML training.")
        df = df.sample(n=ML_SAMPLE_ROWS, seed=42)
    
    pdf = df.to_pandas()
    target_reg = "arr_delay"
    target_clf = "is_delayed"
    
    if target_reg not in pdf.columns or target_clf not in pdf.columns:
        print(f"  Error: Target columns missing.")
        return

    _, feature_cols = _make_preprocessor(pdf)
    if not feature_cols:
        print("  Warning: No features found.")
        return

    X = pdf[feature_cols]
    y_reg = pdf[target_reg]
    y_clf = pdf[target_clf]
    
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf if y_clf.nunique() > 1 else None,
    )
    
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment("Flight_Lab3")

    gold_version = DeltaTable(str(GOLD_PATH / "ml_features")).version()

    regression_models = {
        "DummyMeanRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),
    }

    for name, estimator in regression_models.items():
        with mlflow.start_run(run_name=f"Regression_{name}"):
            print(f"  Training regression model: {name}...")
            preprocessor, _ = _make_preprocessor(pdf)
            model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            model.fit(X_train, y_train_reg)

            pred_reg = model.predict(X_test)
            metrics = _regression_metrics(y_test_reg, pred_reg)

            mlflow.log_param("task", "regression")
            mlflow.log_param("model_type", name)
            mlflow.log_param("features_used", json.dumps(feature_cols))
            mlflow.log_param("training_rows", len(pdf))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("gold_table_version", gold_version)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            _log_feature_importance(model, f"regression_{name}")
            print(
                f"  Regression {name}: "
                f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}"
            )

    classification_models = {
        "DummyMostFrequentClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),
    }

    for name, estimator in classification_models.items():
        with mlflow.start_run(run_name=f"Classification_{name}"):
            print(f"  Training classification model: {name}...")
            preprocessor, _ = _make_preprocessor(pdf)
            model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            model.fit(X_train, y_train_clf)

            pred_clf = model.predict(X_test)
            score_clf = _classification_scores(model, X_test)
            metrics = _classification_metrics(y_test_clf, pred_clf, score_clf)

            mlflow.log_param("task", "classification")
            mlflow.log_param("model_type", name)
            mlflow.log_param("features_used", json.dumps(feature_cols))
            mlflow.log_param("training_rows", len(pdf))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("gold_table_version", gold_version)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            _log_feature_importance(model, f"classification_{name}")
            print(
                f"  Classification {name}: "
                f"accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
                f"ROC-AUC={metrics['roc_auc']:.3f}"
            )

    print("  ML Pipeline completed successfully!")
