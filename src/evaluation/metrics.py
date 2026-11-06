"""Metrics return None when there are no evaluable observations."""

import numpy as np


def rating_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted shapes differ")
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual >= 1) & (actual <= 10)
    errors = predicted[valid] - actual[valid]
    return {"mae": float(np.abs(errors).mean()) if len(errors) else None,
            "rmse": float(np.sqrt(np.square(errors).mean())) if len(errors) else None,
            "rating_evaluated": int(valid.sum())}


def precision_recall_at_k(recommended, relevant, k):
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return None, None
    # Sách trùng không được tính nhiều lần; thiếu đề xuất vẫn chia Precision cho K.
    recommended = list(dict.fromkeys(recommended))[:k]
    hits = len(set(recommended) & relevant)
    return hits / k, hits / len(relevant)
