"""Deterministic statistical routines for reviewer-facing audit scripts."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from .metrics import ipcw_concordance_index


def patient_bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    replicates: int = 1000,
    random_state: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    values = np.asarray(values)
    if values.ndim == 0 or len(values) < 2:
        raise ValueError("patient bootstrap requires at least two patient rows")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(random_state)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = values[rng.integers(0, len(values), size=len(values))]
        draws[index] = float(statistic(sample))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(statistic(values)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "replicates": int(replicates),
        "bootstrap_unit": "patient",
    }


def nadeau_bengio_corrected_ttest(
    paired_differences: np.ndarray, *, test_fraction: float
) -> dict[str, float | int]:
    """Corrected resampled paired t-test for repeated cross-validation."""

    diff = np.asarray(paired_differences, dtype=float)
    if diff.ndim != 1 or len(diff) < 2:
        raise ValueError("paired_differences must contain at least two folds/runs")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie in (0,1)")
    variance = float(np.var(diff, ddof=1))
    train_fraction = 1.0 - test_fraction
    correction = 1.0 / len(diff) + test_fraction / train_fraction
    standard_error = math.sqrt(max(0.0, correction * variance))
    mean = float(np.mean(diff))
    statistic = mean / standard_error if standard_error > 0 else math.copysign(float("inf"), mean)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=len(diff) - 1))
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "t_statistic": float(statistic),
        "degrees_of_freedom": int(len(diff) - 1),
        "p_value_two_sided": p_value,
        "correction_factor": correction,
    }


def mcnemar_exact(
    true_labels: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray
) -> dict[str, float | int]:
    truth = np.asarray(true_labels)
    a = np.asarray(prediction_a)
    b = np.asarray(prediction_b)
    if truth.shape != a.shape or truth.shape != b.shape:
        raise ValueError("truth and paired predictions must have equal shapes")
    a_correct, b_correct = a == truth, b == truth
    b_count = int(np.sum(a_correct & ~b_correct))
    c_count = int(np.sum(~a_correct & b_correct))
    discordant = b_count + c_count
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(b_count, c_count) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "a_correct_b_wrong": b_count,
        "a_wrong_b_correct": c_count,
        "discordant": discordant,
        "p_value_two_sided_exact": float(p_value),
    }


def paired_ipcw_c_index_test(
    times: np.ndarray,
    events: np.ndarray,
    model_risk: np.ndarray,
    comparator_risk: np.ndarray,
    *,
    replicates: int = 1000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Patient-bootstrap variance test for a paired IPCW C-index contrast."""

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    model_risk = np.asarray(model_risk, dtype=float)
    comparator_risk = np.asarray(comparator_risk, dtype=float)
    if (
        times.ndim != 1
        or events.shape != times.shape
        or model_risk.shape != times.shape
        or comparator_risk.shape != times.shape
    ):
        raise ValueError("times, events, and paired risks must be aligned vectors")
    if len(times) < 3 or replicates < 2:
        raise ValueError("at least three patients and two bootstrap replicates are required")

    model_c = ipcw_concordance_index(times, events, model_risk)
    comparator_c = ipcw_concordance_index(times, events, comparator_risk)
    estimate = float(model_c - comparator_c)
    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(replicates):
        indices = rng.integers(0, len(times), size=len(times))
        model_draw = ipcw_concordance_index(
            times[indices], events[indices], model_risk[indices]
        )
        comparator_draw = ipcw_concordance_index(
            times[indices], events[indices], comparator_risk[indices]
        )
        difference = model_draw - comparator_draw
        if np.isfinite(difference):
            differences.append(float(difference))
    if len(differences) < 2:
        raise ValueError("bootstrap draws contain too few comparable event pairs")
    standard_error = float(np.std(differences, ddof=1))
    z_statistic = estimate / standard_error if standard_error > 0 else math.copysign(
        float("inf"), estimate
    )
    return {
        "model_ipcw_c_index": float(model_c),
        "comparator_ipcw_c_index": float(comparator_c),
        "difference": estimate,
        "standard_error": standard_error,
        "z_statistic": float(z_statistic),
        "p_value_two_sided": float(2.0 * norm.sf(abs(z_statistic))),
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_valid": int(len(differences)),
        "bootstrap_unit": "patient",
    }


def decision_curve_net_benefit(
    outcome: np.ndarray, risk: np.ndarray, thresholds: np.ndarray
) -> list[dict[str, float | int]]:
    outcome = np.asarray(outcome, dtype=int)
    risk = np.asarray(risk, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if outcome.shape != risk.shape or not set(np.unique(outcome)).issubset({0, 1}):
        raise ValueError("outcome and risk must be aligned; outcome must be binary")
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("thresholds must lie strictly between zero and one")
    n = len(outcome)
    rows = []
    for threshold in thresholds:
        positive = risk >= threshold
        tp = int(np.sum(positive & (outcome == 1)))
        fp = int(np.sum(positive & (outcome == 0)))
        net_benefit = tp / n - fp / n * threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive": tp,
                "false_positive": fp,
                "net_benefit": float(net_benefit),
            }
        )
    return rows


def e_value_from_risk_ratio(risk_ratio: float) -> float:
    rr = float(risk_ratio)
    if rr <= 0:
        raise ValueError("risk_ratio must be positive")
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + math.sqrt(rr * (rr - 1.0)))
