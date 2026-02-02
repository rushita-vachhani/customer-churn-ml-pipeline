from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


LEAKAGE_COLS = ["Churn Label", "Churn Score", "Churn Reason", "CustomerID"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure target exists
    if "Churn" not in df.columns and "Churn Value" in df.columns:
        df.rename(columns={"Churn Value": "Churn"}, inplace=True)

    # Total Charges cleanup (your dataset uses "Total Charges")
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

    return df


def make_features_and_target(df: pd.DataFrame):
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found. Ensure churn_clean.csv has 'Churn'.")

    y = df["Churn"]
    X = df.drop(columns=["Churn"] + [c for c in LEAKAGE_COLS if c in df.columns])

    # Optional: drop CLTV if you want to be conservative
    if "CLTV" in X.columns:
        X = X.drop(columns=["CLTV"])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return preprocessor
