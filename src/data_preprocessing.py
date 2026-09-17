import re
import string
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"label", "message"}


def clean_text(text):
    """Normalize email text by removing noise and standardizing casing."""
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path):
    """Load the dataset with validation, missing-value handling, and duplicates cleanup."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(path)
    validate_dataset_schema(df)

    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns={"text": "message", "email": "message"})

    if "message" not in df.columns:
        raise ValueError("Dataset must contain a message/text column.")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a label column.")

    df["message"] = df["message"].fillna("")
    df["label"] = df["label"].fillna("ham")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["message", "label"], keep="first")
    df["message"] = df["message"].apply(clean_text)

    return df


def validate_dataset_schema(df):
    """Ensure dataset has the required columns and consistent types."""
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("Dataset schema must include label and message columns.")

    return True


def summarize_dataset(df):
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_distribution": df["label"].value_counts().to_dict(),
    }
    return summary
