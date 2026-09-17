import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import APP_CONFIG, MODELS_DIR, RAW_DATA_DIR
from src.data_preprocessing import load_dataset, summarize_dataset
from src.feature_extraction import build_vectorizer


def train_and_compare_models(csv_path=None, save_path=None):
    csv_path = csv_path or str(RAW_DATA_DIR / "spam_dataset.csv")
    save_path = save_path or str(MODELS_DIR / "best_model.joblib")

    df = load_dataset(csv_path)
    summary = summarize_dataset(df)

    X = df["message"]
    y = df["label"]

    vectorizer = build_vectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    nb_model = MultinomialNB()
    svm_model = LinearSVC(class_weight="balanced", random_state=42)

    models = {
        "MultinomialNB": nb_model,
        "LinearSVM": svm_model,
    }

    results = {}
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring="f1_weighted")
        results[name] = {
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
        }

    # In the full pipeline we train both models separately on the same transformed data.
    # The best model is selected using cross-validated F1 weighted score.
    nb_model.fit(X_tfidf, y)
    svm_model.fit(X_tfidf, y)

    model_selection = {
        "MultinomialNB": results["MultinomialNB"]["cv_f1_mean"],
        "LinearSVM": results["LinearSVM"]["cv_f1_mean"],
    }
    best_model_name = max(model_selection, key=model_selection.get)
    best_model = models[best_model_name]

    if best_model_name == "MultinomialNB":
        model_bundle = {
            "model_name": "MultinomialNB",
            "classifier": best_model,
            "vectorizer": vectorizer,
            "label_values": sorted(y.unique()),
            "dataset_summary": summary,
        }
    else:
        model_bundle = {
            "model_name": "LinearSVM",
            "classifier": best_model,
            "vectorizer": vectorizer,
            "label_values": sorted(y.unique()),
            "dataset_summary": summary,
        }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, save_path)

    metrics = {
        "best_model": best_model_name,
        "training_summary": summary,
        "cross_validation": results,
    }

    with open(str(Path(save_path).with_suffix(".json")), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return model_bundle


if __name__ == "__main__":
    train_and_compare_models()
