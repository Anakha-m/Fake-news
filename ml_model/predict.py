"""
Inference & Prediction Module for Fake News Detection System.
SDG 16: Peace, Justice, and Strong Institutions.

Uses Logistic Regression with TF-IDF features.
Classifies articles directly as REAL or FAKE based on Logistic Regression decision threshold and probabilities.
"""

import os
import sys
import joblib
import numpy as np
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model.preprocess import clean_text

SAVED_MODELS_DIR = os.path.join(CURRENT_DIR, "saved_models")
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl")

_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def load_artifacts():
    """Loads and caches the trained Logistic Regression model and TF-IDF vectorizer."""
    global _CACHED_MODEL, _CACHED_VECTORIZER

    if _CACHED_MODEL is None or _CACHED_VECTORIZER is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Trained model artifacts not found in '{SAVED_MODELS_DIR}'. "
                f"Please run 'python ml_model/train_model.py' first."
            )
        _CACHED_MODEL = joblib.load(MODEL_PATH)
        _CACHED_VECTORIZER = joblib.load(VECTORIZER_PATH)

    return _CACHED_MODEL, _CACHED_VECTORIZER


def predict_news(text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Classifies a news article / headline directly using Logistic Regression.

    Parameters:
        text (str): The news article body or headline text.
        title (str, optional): The article title/headline.

    Returns:
        dict: prediction metadata, probabilities, and explanations.
    """
    full_text = f"{title.strip()} {text.strip()}" if title and title.strip() else (text or "").strip()

    if not full_text:
        return {
            "prediction": "REAL",
            "raw_label": "REAL",
            "confidence": 50.0,
            "real_probability": 0.50,
            "fake_probability": 0.50,
            "is_needs_review": False,
            "verdict_display": "No Text Provided",
            "badge_class": "warning",
            "explanation": "No text was provided for analysis. Please enter a news headline or article.",
            "cleaned_text": ""
        }

    model, vectorizer = load_artifacts()

    # Step 1: Clean text using the exact same preprocessing pipeline used during training
    cleaned = clean_text(full_text)
    if not cleaned:
        cleaned = full_text.lower().strip()

    # Step 2: Transform with TF-IDF Vectorizer
    features = vectorizer.transform([cleaned])

    # Step 3: Compute Probabilities from Logistic Regression
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        # Label 0 -> REAL, Label 1 -> FAKE
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
        prob_fake = float(1.0 / (1.0 + np.exp(-score)))
        prob_real = float(1.0 - prob_fake)
    else:
        pred_val = int(model.predict(features)[0])
        prob_fake = 1.0 if pred_val == 1 else 0.0
        prob_real = 1.0 - prob_fake

    # Step 4: Determine Logistic Regression Prediction and Confidence
    if prob_fake >= prob_real:
        prediction = "FAKE"
        raw_label = "FAKE"
        confidence = round(prob_fake * 100.0, 1)
        is_needs_review = True
        verdict_display = "Potentially Fake News"
        badge_class = "danger"
        explanation = (
            f"The Logistic Regression model classified this content as Fake News "
            f"with {confidence:.1f}% confidence. Queued for Content Writer verification."
        )
    else:
        prediction = "REAL"
        raw_label = "REAL"
        confidence = round(prob_real * 100.0, 1)
        is_needs_review = False
        verdict_display = "Likely Real News"
        badge_class = "success"
        explanation = (
            f"The Logistic Regression model classified this content as Real News "
            f"with {confidence:.1f}% confidence."
        )

    return {
        "prediction": str(prediction),
        "raw_label": str(raw_label),
        "confidence": float(confidence),
        "real_probability": round(float(prob_real), 4),
        "fake_probability": round(float(prob_fake), 4),
        "is_needs_review": bool(is_needs_review),
        "verdict_display": str(verdict_display),
        "badge_class": str(badge_class),
        "explanation": str(explanation),
        "cleaned_text": str(cleaned)
    }


if __name__ == "__main__":
    test_cases = [
        "The government has reportedly announced a new nationwide welfare scheme under which every adult citizen will receive 10,000 per month directly into their bank account. The scheme is expected to begin from next month, according to a message circulating on social media.",
        "NASA James Webb Space Telescope discovers earliest distant galaxy formed after Big Bang with spectroscopic redshift measurements."
    ]

    print("\n--- Testing Calibrated Prediction Engine ---")
    for t in test_cases:
        res = predict_news(t)
        print(f"\nText: {t[:65]}...")
        print(f"Prediction: {res['prediction']} ({res['verdict_display']})")
        print(f"Confidence: {res['confidence']}% | Real: {res['real_probability']*100:.1f}% | Fake: {res['fake_probability']*100:.1f}%")
        print(f"Needs Review: {res['is_needs_review']}")
