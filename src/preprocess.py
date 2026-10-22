"""Preprocessing utilities for the IMDB sentiment analysis project.

The original notebook performed a series of ad‑hoc cleaning steps.  Here we
expose the same logic as reusable functions, fixing the two critical bugs
identified earlier:

1. **HTML tag removal must happen before punctuation removal.**
2. **Stop‑word removal should filter tokens instead of doing a substring
   replace**, which corrupts words that merely contain a stop‑word.

The public API is:

* ``load_dataset(csv_path)`` – returns a pandas ``DataFrame`` with duplicate
  rows dropped.
* ``clean_review(text)`` – runs the full preprocessing pipeline on a single
  review string.
* ``preprocess_dataframe(df)`` – applies ``clean_review`` to the ``review``
  column and returns the cleaned ``DataFrame``.
"""

from __future__ import annotations

import re
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are available (they were
# downloaded in the original notebook).
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

_stop_words_set = set(stopwords.words("english"))
_ps = PorterStemmer()


def _remove_urls(text: str) -> str:
    """Strip http/https URLs from *text* using a simple regex."""
    return re.sub(r"http\S+", "", text)


def _remove_html_tags(text: str) -> str:
    """Remove any HTML tags (e.g. ``<br/>``) from *text*.

    The pattern ``<.*?>`` is non‑greedy and matches the smallest possible tag.
    """
    return re.sub(r"<.*?>", "", text)


def _remove_punctuations(text: str) -> str:
    """Delete characters that are not letters, numbers or whitespace."""
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


def _remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Filter tokens that appear in the NLTK English stopword list."""
    return [tok for tok in tokens if tok not in _stop_words_set]


def _stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Apply Porter stemming to each token and return the stemmed list."""
    return [_ps.stem(tok) for tok in tokens]


def clean_review(text: str) -> str:
    """Run the full cleaning pipeline on a single review.

    Order of operations (important!):
    1. Lower‑case
    2. URL removal
    3. HTML tag removal
    4. Punctuation removal
    5. Tokenisation
    6. Stop‑word filtering
    7. Stemming
    8. Re‑join tokens with a single space
    """
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_html_tags(text)
    text = _remove_punctuations(text)
    tokens = word_tokenize(text)
    tokens = _remove_stopwords(tokens)
    tokens = _stem_tokens(tokens)
    return " ".join(tokens)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV, drop duplicate rows and return the ``DataFrame``.

    The original dataset contains two columns: ``review`` and ``sentiment``.
    """
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`clean_review` to the ``review`` column in‑place and return *df*."""
    df = df.copy()
    df["review"] = df["review"].apply(clean_review)
    return df


__all__ = [
    "load_dataset",
    "clean_review",
    "preprocess_dataframe",
]
