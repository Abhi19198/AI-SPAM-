import pandas as pd

from src.data_preprocessing import clean_text, validate_dataset_schema
from src.predict import predict_email


def test_validate_dataset_schema_accepts_required_columns():
    df = pd.DataFrame({
        "label": ["spam", "ham", "spam"],
        "message": ["Win cash now", "Project update", "Free prize"],
    })

    result = validate_dataset_schema(df)
    assert result is True


def test_clean_text_removes_noise_and_normalizes():
    text = "  FREE!!! $$$ Click https://example.com now  "
    cleaned = clean_text(text)
    assert "free" in cleaned
    assert "click" in cleaned
    assert "https" not in cleaned
    assert cleaned.strip() == cleaned


def test_predict_email_returns_expected_shape():
    result = predict_email("Congratulations, you have won a free prize.")
    assert "prediction" in result
    assert "confidence" in result
    assert result["prediction"] in {"spam", "ham"}
    assert 0 <= result["confidence"] <= 100
