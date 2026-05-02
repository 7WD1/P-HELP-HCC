"""Evaluation metrics for eight-class HCC survival prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, top)]))


def ordinal_risk_from_proba(proba: np.ndarray) -> np.ndarray:
    expected_class = proba @ np.arange(proba.shape[1])
    return -expected_class


def harrell_c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j]:
                continue
            if events[i] == 1 and times[i] < times[j]:
                comparable += 1
                concordant += float(risks[i] > risks[j]) + 0.5 * float(risks[i] == risks[j])
            elif events[j] == 1 and times[j] < times[i]:
                comparable += 1
                concordant += float(risks[j] > risks[i]) + 0.5 * float(risks[i] == risks[j])
    return float(concordant / comparable) if comparable else float("nan")


def classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    times: np.ndarray | None = None,
    events: np.ndarray | None = None,
) -> dict[str, float]:
    y_pred = np.argmax(proba, axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "top2_accuracy": top_k_accuracy(y_true, proba, 2),
        "top3_accuracy": top_k_accuracy(y_true, proba, 3),
    }
    if times is not None and events is not None:
        out["c_index"] = harrell_c_index(times, events, ordinal_risk_from_proba(proba))
    return out

