# AI-Powered Spam Email Classifier

## Project Overview
This project builds a production-style spam email detector using supervised learning. It processes raw email messages, vectorizes text with TF-IDF, compares a Multinomial Naive Bayes model with a Linear SVM, and exposes a Flask API plus a modern dashboard UI.

## Problem Statement
Email inboxes receive both legitimate and malicious messages. A good spam filter should classify emails with reasonable accuracy while handling class imbalance, noisy text, and generalization to unseen messages.

## Features
- Dataset validation and preprocessing pipeline
- TF-IDF vectorization
- Multinomial Naive Bayes and Linear SVM comparison
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Flask REST API with validation and error handling
- SQLite database for prediction history
- Premium dark-themed frontend
- Tests with pytest
- Deployment-ready structure with Docker support

## Architecture Diagram
```text
CSV Dataset
   -> Validation + Cleaning
   -> TF-IDF Vectorizer
   -> Train/Test Split
   -> Model Training (NB and SVM)
   -> Evaluation + Model Selection
   -> Save best pipeline to joblib
   -> Flask API + UI consumes saved model
```

## Technology Stack
- Python
- Pandas
- NumPy
- scikit-learn
- Flask
- SQLite
- JavaScript
- HTML/CSS
- pytest
- joblib

## Project Structure
```text
spam-email-classifier/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── config.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── instance/
│   └── database.db
└── Dockerfile
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Dataset Setup
Place your dataset in `data/raw/` as a CSV file with at least these columns:
- `label`
- `message`

Example:
```csv
label,message
spam,Win a free vacation now
ham,Project meeting scheduled for tomorrow
```

## Model Training
```bash
python src/train.py
```

## API Usage
```bash
python app.py
```
Then open:
- http://localhost:5000/
- http://localhost:5000/api/health

## Deployment Instructions
```bash
docker build -t spam-email-classifier .
docker run -p 5000:5000 spam-email-classifier
```

## Limitations
- This is a classic text-classification approach and not a replacement for advanced transformer models.
- Confidence is an estimate and should not be treated as certainty.
- Performance depends on dataset quality and class balance.

## Future Improvements
- Add BERT or DistilBERT models
- Use better dataset balancing and augmentation
- Add user feedback loop
- Deploy with PostgreSQL and Redis

## Contribution Guidelines
1. Fork the repository.
2. Create a feature branch.
3. Add tests for new behavior.
4. Validate with pytest.
5. Open a pull request with a clear explanation.

## Learning Notes
This project demonstrates supervised learning, TF-IDF, Naive Bayes, SVM, model validation, confidence estimation, and production-style deployment patterns.
