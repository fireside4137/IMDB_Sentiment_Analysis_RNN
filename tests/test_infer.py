from sklearn.feature_extraction.text import TfidfVectorizer

from src.infer import predict_sentiment
from src.model import RNN


def test_predict_sentiment() -> None:
    corpus = ["great movie", "terrible film"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    model = RNN(input_size=len(vectorizer.get_feature_names_out()))
    model.eval()

    label, prob = predict_sentiment("great movie", model, vectorizer)
    assert label in {"positive", "negative"}
    assert 0.0 <= prob <= 1.0
