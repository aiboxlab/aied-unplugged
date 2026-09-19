from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=False):
        row, column = index.get(true), index.get(pred)
        if row is not None and column is not None:
            matrix[row, column] += 1
    return matrix


def quadratic_weighted_kappa(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    observed = confusion(y_true, y_pred, labels).astype(float)
    n = len(labels)
    if n < 2:
        return 1.0 if observed.sum() else 0.0

    grid = np.arange(n)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (n - 1) ** 2

    total = observed.sum()
    if total == 0:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total

    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    # A constant prediction makes the denominator zero; score it by exactness.
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - numerator / denominator


def per_class_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> dict[str, float]:
    y_true, y_pred = list(y_true), list(y_pred)
    scores: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p != label)
        denominator = 2 * tp + fp + fn
        scores[label] = (2 * tp / denominator) if denominator else 0.0
    return scores


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    if not len(labels):
        return 0.0
    return float(np.mean(list(per_class_f1(y_true, y_pred, labels).values())))


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    y_true = list(y_true)
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((true - pred) ** 2)))
