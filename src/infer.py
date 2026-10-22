"""Inference script for the IMDB sentiment analysis project.

Usage:
    python -m src.infer "This movie was absolutely wonderful and thrilling!"
    python -m src.infer --model models/rnn_model.pt "Great film"
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.model import RNN
from src.preprocess import clean_review


def load_artifacts(
    model_path: pathlib.Path, vectorizer_path: pathlib.Path
) -> Tuple[RNN, TfidfVectorizer]:
    """Load the trained RNN model and TF-IDF vectorizer from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train the model first using `python -m src.train`."
        )
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found at {vectorizer_path}. "
            "Please train the model first using `python -m src.train`."
        )

    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    input_dim = len(vectorizer.get_feature_names_out())

    model = RNN(input_size=input_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return model, vectorizer


def predict_sentiment(
    text: str, model: RNN, vectorizer: TfidfVectorizer
) -> Tuple[str, float]:
    """Predict sentiment and probability for a given review text."""
    cleaned = clean_review(text)
    sparse_vec = vectorizer.transform([cleaned])
    dense_vec = sparse_vec.toarray().copy()
    tensor_input = torch.from_numpy(dense_vec).float().unsqueeze(1)

    with torch.no_grad():
        logits = model(tensor_input).squeeze()
        prob = torch.sigmoid(logits).item()

    label = "positive" if prob >= 0.5 else "negative"
    return label, prob


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict sentiment of movie review text."
    )
    parser.add_argument("review", type=str, help="Movie review text to analyze")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path("models/rnn_model.pt"),
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--vectorizer",
        type=pathlib.Path,
        default=pathlib.Path("models/tfidf_vectorizer.pkl"),
        help="Path to fitted TF-IDF vectorizer",
    )
    args = parser.parse_args()

    # Fallback to root directory if models/ doesn't have them yet
    model_path = args.model
    vectorizer_path = args.vectorizer
    if not model_path.exists() and pathlib.Path("rnn_model.pt").exists():
        model_path = pathlib.Path("rnn_model.pt")
    if not vectorizer_path.exists() and pathlib.Path("tfidf_vectorizer.pkl").exists():
        vectorizer_path = pathlib.Path("tfidf_vectorizer.pkl")

    try:
        model, vectorizer = load_artifacts(model_path, vectorizer_path)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    label, prob = predict_sentiment(args.review, model, vectorizer)
    confidence = prob if label == "positive" else 1.0 - prob
    print(f"Sentiment:  {label.upper()}")
    print(f"Confidence: {confidence * 100:.2f}% (prob={prob:.4f})")


if __name__ == "__main__":
    main()
