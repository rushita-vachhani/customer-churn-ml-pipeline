from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/churn_model.joblib"

app = FastAPI(title="Churn Prediction API")

model = None


class CustomerFeatures(BaseModel):
    # Keep this flexible: accept arbitrary fields
    # Will pass into a DataFrame for the pipeline.
    features: dict


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerFeatures):
    # payload.features is a dict of column->value
    X = pd.DataFrame([payload.features])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return {"churn_probability": proba, "churn_prediction": pred}
