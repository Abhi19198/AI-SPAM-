import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.data_preprocessing import load_dataset
from src.feature_extraction import build_vectorizer


def evaluate_model(csv_path=None, model_path=None):
    csv_path = csv_path or str(RAW_DATA_DIR / "spam_dataset.csv")
    model_path = model_path or str(MODELS_DIR / "best_model.joblib")

    df = load_dataset(csv_path)
    X = df["message"]
    y = df["label"]

    model_bundle = joblib.load(model_path)
    vectorizer = model_bundle["vectorizer"]
    classifier = model_bundle["classifier"]

    X_tfidf = vectorizer.transform(X)
    preds = classifier.predict(X_tfidf)

    metrics = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds, output_dict=True, zero_division=0),
    }

    return metrics
