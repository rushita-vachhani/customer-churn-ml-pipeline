# dashboard/app.py
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("Customer Churn Predictor")
st.write("Enter customer details to estimate churn risk.")

model = joblib.load(MODEL_PATH)

# Minimal input set (you can expand)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=1000.0)
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)

if st.button("Predict"):
    row = {
        "Contract": contract,
        "Tenure Months": tenure,
        "Monthly Charges": monthly,
        "Total Charges": total,
        "Paperless Billing": paperless,
        "Payment Method": payment,
    }
    X = pd.DataFrame([row])

    expected_features = list(model.feature_names_in_)
    df_template = pd.read_csv(DATA_PATH)
    for col in expected_features:
        if col not in X.columns:
            if pd.api.types.is_numeric_dtype(df_template[col]):
                X[col] = 0
            else:
                X[col] = "Unknown"
    X = X[expected_features]

    proba = float(model.predict_proba(X)[:, 1][0])
    threshold = 0.35
    st.metric("Churn Probability", f"{proba:.2%}")
    st.write("Prediction:", "Churn" if proba >= threshold else "No Churn")
