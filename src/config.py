import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
INSTANCE_DIR = BASE_DIR / "instance"
DB_PATH = INSTANCE_DIR / "database.db"

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


def get_env_or_default(key, default):
    return os.getenv(key, default)


APP_CONFIG = {
    "APP_NAME": get_env_or_default("APP_NAME", "AI Spam Email Classifier"),
    "SECRET_KEY": get_env_or_default("SECRET_KEY", "dev-secret-key"),
    "MODEL_PATH": get_env_or_default("MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    "MAX_INPUT_LENGTH": int(get_env_or_default("MAX_INPUT_LENGTH", "2000")),
    "RATE_LIMIT_PER_MINUTE": int(get_env_or_default("RATE_LIMIT_PER_MINUTE", "60")),
    "ENABLE_HISTORY": get_env_or_default("ENABLE_HISTORY", "true").lower() == "true",
    "FLASK_ENV": get_env_or_default("FLASK_ENV", "development"),
}
