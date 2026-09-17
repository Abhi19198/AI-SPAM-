import logging
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest

from src.config import APP_CONFIG, BASE_DIR
from src.database import get_prediction_stats, init_db, record_prediction
from src.predict import load_model_bundle, predict_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
app.config["SECRET_KEY"] = APP_CONFIG["SECRET_KEY"]

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{APP_CONFIG['RATE_LIMIT_PER_MINUTE']}/minute"],
    storage_uri="memory://",
)

MODEL = None


def load_model():
    global MODEL
    try:
        MODEL = load_model_bundle(APP_CONFIG["MODEL_PATH"])
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.exception("Model load failed: %s", exc)
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "app": APP_CONFIG["APP_NAME"],
        "model_loaded": MODEL is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/predict", methods=["POST"])
@limiter.limit("10/minute")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        email_text = payload.get("email_text")

        if not isinstance(email_text, str):
            return jsonify({"error": "Request must include a string field named 'email_text'."}), 400

        text = email_text.strip()
        if not text:
            return jsonify({"error": "Email text cannot be empty."}), 400

        if len(text) > APP_CONFIG["MAX_INPUT_LENGTH"]:
            return jsonify({"error": "Input too long."}), 400

        prediction = predict_email(text, APP_CONFIG["MODEL_PATH"])
        explanation = [
            "TF-IDF weightings identified words with strong spam signals.",
            "This is a probability-based model estimate, not proof of intent or fact.",
        ]

        record_prediction(prediction["prediction"], prediction["confidence"], prediction["model"])

        return jsonify({
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "model": prediction["model"],
            "explanation": explanation,
        }), 200
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        logger.exception("Model file missing: %s", exc)
        return jsonify({"error": "Model is not available. Please train the model first."}), 500
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": "Prediction failed due to an internal error."}), 500


@app.route("/api/batch-predict", methods=["POST"])
@limiter.limit("5/minute")
def batch_predict():
    try:
        payload = request.get_json(silent=True) or {}
        emails = payload.get("emails")

        if not isinstance(emails, list) or not emails:
            return jsonify({"error": "Request body must contain a non-empty 'emails' list."}), 400

        result = []
        for item in emails:
            if not isinstance(item, str):
                return jsonify({"error": "Each item in 'emails' must be a string."}), 400
            clean_item = item.strip()
            if not clean_item:
                return jsonify({"error": "One or more email texts are empty."}), 400
            prediction = predict_email(clean_item, APP_CONFIG["MODEL_PATH"])
            result.append({
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "model": prediction["model"],
            })

        return jsonify({"results": result}), 200
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        return jsonify({"error": "Batch prediction failed."}), 500


@app.get("/api/stats")
def stats():
    stats_data = get_prediction_stats()
    return jsonify({
        "total_predictions": stats_data["total_predictions"],
        "spam_count": stats_data["spam_count"],
        "ham_count": stats_data["ham_count"],
        "status": "ok",
        "note": "Prediction history stores only metadata and not raw email content.",
    })


@app.before_request
def initialize_app_state():
    if not os.path.exists("instance"):
        os.makedirs("instance", exist_ok=True)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Request body is too large."}), 413


if __name__ == "__main__":
    init_db()
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=APP_CONFIG["FLASK_ENV"] == "development")
