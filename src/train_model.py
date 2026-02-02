from __future__ import annotations

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_data, clean_data, make_features_and_target, build_preprocessor


DATA_PATH = "data/processed/churn_clean.csv"
MODEL_PATH = "models/churn_model.joblib"


def train():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = make_features_and_target(df)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest with imbalance handling
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    # Evaluate using probabilities + threshold (better for churn)
    y_proba = clf.predict_proba(X_test)[:, 1]

    threshold = 0.35
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", round(roc_auc, 4))
    print(f"\nClassification Report (threshold={threshold}):\n",
          classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
