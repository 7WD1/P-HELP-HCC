"""Phase P prediction-to-observed-outcome feedback controller.

Implements the dual-rule controller of paper Eq.~(16) and Table VII:
streaming residual triggers ``e_soft=0.18`` (recalibrate) / ``e_hard=0.32``
(retrain), and per-prediction Shannon-entropy abstention bands
``p_soft=0.65`` (soft-abstain) / ``p_hard=0.85`` (hard-abstain).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def phase_p_residuals(
    probabilities: np.ndarray,
    true_classes: np.ndarray,
    events: np.ndarray,
    *,
    classification_calibration_mix: float = 0.5,
) -> np.ndarray:
    """Return the per-row replay residual used by Phase P.

    This is the vector form of :meth:`ParallelController.observe`.  Keeping
    one implementation for checkpoint/candidate selection and replay avoids
    a silent mismatch between the training and monitoring paths.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    true_classes = np.asarray(true_classes, dtype=int)
    events = np.asarray(events, dtype=int)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if true_classes.shape != (len(probabilities),) or events.shape != (len(probabilities),):
        raise ValueError("true_classes and events must align with probabilities")
    if np.any((true_classes < 0) | (true_classes >= probabilities.shape[1])):
        raise ValueError("true_classes contain an out-of-range class index")
    row_sum = probabilities.sum(axis=1)
    if np.any(probabilities < 0) or not np.allclose(row_sum, 1.0, atol=1e-6):
        raise ValueError("probabilities must be non-negative and sum to one")
    target = np.eye(probabilities.shape[1], dtype=float)[true_classes]
    confidence_error = 1.0 - probabilities[np.arange(len(probabilities)), true_classes]
    calibration_error = np.square(probabilities - target).sum(axis=1)
    return confidence_error + float(classification_calibration_mix) * calibration_error


def phase_p_residual_score(
    probabilities: np.ndarray,
    true_classes: np.ndarray,
    events: np.ndarray,
    *,
    classification_calibration_mix: float = 0.5,
) -> float:
    """Mean Phase-P replay residual; lower values are preferred."""

    return float(
        np.mean(
            phase_p_residuals(
                probabilities,
                true_classes,
                events,
                classification_calibration_mix=classification_calibration_mix,
            )
        )
    )


def _reverse_km_censoring(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse Kaplan--Meier estimate of the censoring survival function."""

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    unique_times = np.unique(times)
    survival = 1.0
    timeline = [0.0]
    values = [1.0]
    for time in unique_times:
        at_risk = float(np.sum(times >= time))
        if at_risk <= 0:
            continue
        censorings = float(np.sum((times == time) & (events == 0)))
        if censorings:
            survival *= max(0.0, 1.0 - censorings / at_risk)
        timeline.append(float(time))
        values.append(float(survival))
    return np.asarray(timeline), np.asarray(values)


def _left_continuous_lookup(time: float, timeline: np.ndarray, values: np.ndarray) -> float:
    index = int(np.searchsorted(timeline, time, side="left") - 1)
    return float(values[max(0, min(index, len(values) - 1))])


def phase_p_ipcw_residual_weights(
    train_classes: np.ndarray,
    train_times: np.ndarray,
    train_events: np.ndarray,
    validation_classes: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_events: np.ndarray,
    cutpoints: list[float],
    *,
    classification_calibration_mix: float = 0.5,
    residual_strength: float = 1.0,
    minimum_positive_weight: float = 0.25,
    maximum_weight: float = 4.0,
    censoring_floor: float = 1e-3,
) -> tuple[np.ndarray, dict[str, float | list[float]]]:
    """Construct Phase-P-informed IPCW training weights.

    A training row contributes a definitive class target only after an event
    or after being observed through the final class boundary.  Its IPCW term
    is multiplied by the validation-stream residual of its class.  Rows
    censored before the final boundary receive a numerical epsilon so sklearn
    retains a stable class vocabulary while their classification loss is
    effectively zero.
    """

    train_classes = np.asarray(train_classes, dtype=int)
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=int)
    validation_classes = np.asarray(validation_classes, dtype=int)
    validation_probabilities = np.asarray(validation_probabilities, dtype=float)
    validation_events = np.asarray(validation_events, dtype=int)
    n_train = len(train_classes)
    if train_times.shape != (n_train,) or train_events.shape != (n_train,):
        raise ValueError("training time/event arrays must align with train_classes")
    if not cutpoints:
        raise ValueError("cutpoints are required for censoring-aware sample weights")
    residuals = phase_p_residuals(
        validation_probabilities,
        validation_classes,
        validation_events,
        classification_calibration_mix=classification_calibration_mix,
    )
    n_classes = validation_probabilities.shape[1]
    global_residual = float(np.mean(residuals))
    class_residuals = np.full(n_classes, global_residual, dtype=float)
    for class_index in range(n_classes):
        mask = validation_classes == class_index
        if np.any(mask):
            class_residuals[class_index] = float(np.mean(residuals[mask]))

    timeline, censoring_survival = _reverse_km_censoring(train_times, train_events)
    horizon = float(max(cutpoints))
    matured = (train_events == 1) | (train_times >= horizon)
    ipcw = np.zeros(n_train, dtype=float)
    for index in np.flatnonzero(matured):
        evaluation_time = min(float(train_times[index]), horizon)
        g_before = max(
            _left_continuous_lookup(evaluation_time, timeline, censoring_survival),
            float(censoring_floor),
        )
        ipcw[index] = 1.0 / g_before
    residual_factor = 1.0 + float(residual_strength) * class_residuals[train_classes]
    raw = ipcw * residual_factor
    positive = raw > 0
    if not np.any(positive):
        raw = np.ones(n_train, dtype=float)
        positive = np.ones(n_train, dtype=bool)
    raw[positive] = np.clip(raw[positive], minimum_positive_weight, maximum_weight)
    zero_fraction = float(np.mean(~positive))
    raw[~positive] = 1e-6
    raw /= max(float(np.mean(raw)), 1e-12)
    diagnostics: dict[str, float | list[float]] = {
        "mean": float(np.mean(raw)),
        "minimum": float(np.min(raw)),
        "maximum": float(np.max(raw)),
        "early_censored_fraction": zero_fraction,
        "validation_residual": global_residual,
        "class_residuals": class_residuals.tolist(),
    }
    return raw, diagnostics


@dataclass
class ParallelController:
    soft_error_threshold: float = 0.18
    hard_error_threshold: float = 0.32
    abstention_entropy_soft: float = 0.65
    abstention_entropy_hard: float = 0.85
    online_learning_rate: float = 5e-3
    proximal_weight: float = 1e-2
    monitor_window: int = 30
    retrain_buffer: int = 200
    classification_calibration_mix: float = 0.5
    errors: deque[float] = field(default_factory=deque)

    def observe(
        self,
        probabilities: np.ndarray,
        true_class: int,
        event: int | None = None,
    ) -> dict[str, float | str]:
        probabilities = np.asarray(probabilities, dtype=float).reshape(1, -1)
        err = float(
            phase_p_residuals(
                probabilities,
                np.asarray([true_class], dtype=int),
                np.asarray([0 if event is None else event], dtype=int),
                classification_calibration_mix=self.classification_calibration_mix,
            )[0]
        )
        self.errors.append(err)
        while len(self.errors) > self.monitor_window:
            self.errors.popleft()
        avg = sum(self.errors) / max(1, len(self.errors))
        if avg > self.hard_error_threshold:
            action = "full_retrain"
        elif avg > self.soft_error_threshold:
            action = "soft_update"
        else:
            action = "no_update"
        return {"error": err, "streaming_error": avg, "action": action}

    def abstain(self, probabilities: np.ndarray) -> dict[str, float | str | bool]:
        """Per-prediction Shannon-entropy abstention rule (paper Table VII).

        Returns ``hard_abstain`` (entropy >= e_hard, never surface),
        ``soft_abstain`` (entropy >= e_soft, surface with caveat), or
        ``serve`` (entropy < e_soft, surface normally).
        """

        h = _entropy(probabilities)
        if h >= self.abstention_entropy_hard:
            decision = "hard_abstain"
        elif h >= self.abstention_entropy_soft:
            decision = "soft_abstain"
        else:
            decision = "serve"
        return {"entropy": h, "decision": decision, "abstain": decision != "serve"}

    def soft_update_fusion_weights(
        self,
        weights: np.ndarray,
        gradient: np.ndarray,
        anchor_weights: np.ndarray,
    ) -> np.ndarray:
        """Apply the paper's proximal online step to fusion weights."""

        updated = (
            weights
            - self.online_learning_rate * gradient
            - self.proximal_weight * (weights - anchor_weights)
        )
        updated = np.maximum(updated, 0.0)
        total = updated.sum()
        if total <= 0:
            return np.full_like(updated, 1.0 / len(updated))
        return updated / total

    def update_or_retrain_decision(self, current_weights: np.ndarray, gradient: np.ndarray, anchor_weights: np.ndarray):
        avg = sum(self.errors) / max(1, len(self.errors))
        if avg > self.hard_error_threshold:
            return "full_retrain", current_weights
        if avg > self.soft_error_threshold:
            return "soft_update", self.soft_update_fusion_weights(current_weights, gradient, anchor_weights)
        return "no_update", current_weights
