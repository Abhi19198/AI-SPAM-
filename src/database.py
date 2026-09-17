import sqlite3
from pathlib import Path

from src.config import APP_CONFIG, DB_PATH


def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_name TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def record_prediction(prediction, confidence, model_name):
    if not APP_CONFIG["ENABLE_HISTORY"]:
        return False

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO prediction_history (prediction, confidence, model_name, timestamp) VALUES (?, ?, ?, datetime('now'))",
        (str(prediction), float(confidence), str(model_name)),
    )
    conn.commit()
    conn.close()
    return True


def get_prediction_stats():
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_predictions,
            SUM(CASE WHEN prediction = 'spam' THEN 1 ELSE 0 END) AS spam_count,
            SUM(CASE WHEN prediction = 'ham' THEN 1 ELSE 0 END) AS ham_count
        FROM prediction_history
        """
    ).fetchone()
    conn.close()

    return {
        "total_predictions": int(row["total_predictions"] or 0),
        "spam_count": int(row["spam_count"] or 0),
        "ham_count": int(row["ham_count"] or 0),
    }
