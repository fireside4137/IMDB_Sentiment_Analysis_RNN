"""Utility functions for training the IMDB sentiment model.

This module contains:

* ``compute_metrics`` – given true labels and model predictions, returns
  accuracy, precision, recall and F1 score.
* ``EarlyStopping`` – a simple early‑stopping helper that stops training when
  a monitored metric does not improve for ``patience`` epochs.
* ``get_device`` – chooses the best available torch device (CUDA, MPS or CPU).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dictionary with ``accuracy``, ``precision``, ``recall`` and ``f1``.

    ``y_pred`` should contain binary predictions (0 or 1).  ``y_true`` may be
    either ``int`` or ``float`` – it is flattened to a 1‑D ``np.ndarray``.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


class EarlyStopping:
    """Early‑stopping helper.

    Parameters
    ----------
    patience: int
        Number of epochs with no improvement after which training will be
        stopped.
    mode: str, optional
        ``'min'`` means lower metric is better (e.g., loss), ``'max'`` means
        higher is better (e.g., accuracy).  Defaults to ``'min'``.
    delta: float, optional
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, mode: str = "min", delta: float = 0.0):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """Update the early‑stopping state with ``metric``.

        Returns ``True`` if training should stop, ``False`` otherwise.
        """
        score = metric if self.mode == "max" else -metric
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def get_device() -> torch.device:
    """Return the best torch device available.

    Preference order: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["compute_metrics", "EarlyStopping", "get_device"]
