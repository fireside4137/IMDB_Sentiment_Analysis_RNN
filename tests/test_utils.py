import numpy as np
import pytest

from src.utils import EarlyStopping, compute_metrics


def test_compute_metrics_binary() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["precision"] == pytest.approx(1.0)  # TP / (TP+FP) = 2/2
    assert metrics["recall"] == pytest.approx(0.6666666667)
    assert metrics["f1"] == pytest.approx(0.8)


def test_early_stopping_patience() -> None:
    stopper = EarlyStopping(patience=2, mode="min")
    # First call sets best_score
    assert not stopper(0.5)
    # No improvement, counter increments
    assert not stopper(0.6)
    # Exceeds patience -> should stop
    assert stopper(0.7)
