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


def censoring_survival_before(
    times: np.ndarray, events: np.ndarray
) -> np.ndarray:
    """Kaplan--Meier censoring survival immediately before each observed time."""

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if times.ndim != 1 or events.shape != times.shape:
        raise ValueError("times and events must be aligned one-dimensional arrays")
    if np.any(times < 0) or not set(np.unique(events)).issubset({0, 1}):
        raise ValueError("times must be non-negative and events must be binary")

    survival = 1.0
    before_by_time: dict[float, float] = {}
    for time in np.unique(times):
        at_risk = int(np.sum(times >= time))
        before_by_time[float(time)] = survival
        censored = int(np.sum((times == time) & (events == 0)))
        if at_risk:
            survival *= 1.0 - censored / at_risk
    return np.asarray([before_by_time[float(time)] for time in times], dtype=float)


def ipcw_concordance_index(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    *,
    tau: float | None = None,
    epsilon: float = 1e-8,
) -> float:
    """IPCW pair-weighted survival concordance.

    Each comparable pair is anchored by an observed event and weighted by
    the inverse square of the censoring survival immediately before that
    event.  This is the censoring-aware concordance used by the audit path;
    no class labels are imputed for immature censored observations.
    """

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    if times.ndim != 1 or events.shape != times.shape or risks.shape != times.shape:
        raise ValueError("times, events, and risks must be aligned vectors")
    g_before = censoring_survival_before(times, events)
    horizon = float(np.max(times) if tau is None else tau)
    numerator = 0.0
    denominator = 0.0
    for index in range(len(times)):
        if events[index] != 1 or times[index] > horizon or g_before[index] <= epsilon:
            continue
        comparators = times > times[index]
        pair_count = int(np.sum(comparators))
        if pair_count == 0:
            continue
        weight = 1.0 / (g_before[index] ** 2)
        compared_risks = risks[comparators]
        concordant = np.sum(risks[index] > compared_risks)
        tied = np.sum(risks[index] == compared_risks)
        numerator += weight * (float(concordant) + 0.5 * float(tied))
        denominator += weight * pair_count
    return float(numerator / denominator) if denominator else float("nan")


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
        risks = ordinal_risk_from_proba(proba)
        out["c_index"] = ipcw_concordance_index(times, events, risks)
        out["harrell_c_index_unweighted"] = harrell_c_index(times, events, risks)
    return out
