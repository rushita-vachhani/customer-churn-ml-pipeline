from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-style ML inference API using FastAPI + scikit-learn pipeline",
    version="1.0"
)

model = None
expected_features = []


class PredictionRequest(BaseModel):
    features: dict

numeric_features = []
categorical_features = []

@app.on_event("startup")
def load_model():
    global model, expected_features, numeric_features, categorical_features

    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found")

    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        raise RuntimeError("Model does not contain feature schema")

    df = pd.read_csv(DATA_PATH)

    for col in expected_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)


@app.get("/")
def root():
    return {"message": "Churn Prediction API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/schema")
def get_schema():
    """
    Shows the features expected by the model.
    Useful for debugging and clients.
    """
    return {"expected_features": expected_features}


@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        X = pd.DataFrame([request.features])

        for col in expected_features:
            if col not in X.columns:
                if col in numeric_features:
                    X[col] = 0
                else:
                    X[col] = "Unknown"

        X = X[expected_features]

        proba = float(model.predict_proba(X)[0][1])
        prediction = int(proba >= 0.5)

        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))