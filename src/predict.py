import os

import joblib
import numpy as np

from src.config import APP_CONFIG
from src.data_preprocessing import clean_text


def load_model_bundle(model_path=None):
    model_path = model_path or APP_CONFIG["MODEL_PATH"]
    model_path = str(model_path)

    if not os.path.exists(model_path):
        default_dataset = os.path.join("data", "raw", "spam_dataset.csv")
        if os.path.exists(default_dataset):
            from src.train import train_and_compare_models

            train_and_compare_models(default_dataset, model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    return bundle


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def predict_email(text, model_path=None):
    if not isinstance(text, str):
        raise ValueError("Email text must be a string.")

    clean = clean_text(text)
    if not clean:
        raise ValueError("Email text cannot be empty after cleaning.")

    bundle = load_model_bundle(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    X = vectorizer.transform([clean])
    predicted_label = classifier.predict(X)[0]
    classes = list(classifier.classes_)

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X)[0]
        confidence = float(max(probabilities) * 100)
        prob_map = {str(label): round(float(prob), 4) for label, prob in zip(classes, probabilities)}
    else:
        decision_values = np.asarray(classifier.decision_function(X)[0], dtype=float)
        if decision_values.ndim == 0:
            decision_values = np.array([float(decision_values)])

        if len(decision_values) == 2:
            margin = float(decision_values[0])
            score_strength = _sigmoid(margin)
            confidence = float(score_strength * 100 if predicted_label == classes[1] else (1 - score_strength) * 100)
            prob_map = {
                str(classes[0]): round(float(1 - score_strength), 4),
                str(classes[1]): round(float(score_strength), 4),
            }
        else:
            index = int(np.argmax(decision_values))
            confidence = float(_sigmoid(float(decision_values[index])) * 100)
            prob_map = {str(label): round(float(_sigmoid(float(value))), 4) for label, value in zip(classes, decision_values)}

    return {
        "prediction": str(predicted_label),
        "confidence": round(confidence, 2),
        "model": bundle.get("model_name", "MultinomialNB"),
        "probabilities": prob_map,
        "processed_text": clean,
    }
