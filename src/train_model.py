from __future__ import annotations

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression
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

    # Choose model (RF tends to work well for tabular data)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", round(roc_auc, 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
