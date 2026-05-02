"""Phase P virtual-real feedback controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParallelController:
    soft_error_threshold: float = 0.18
    hard_error_threshold: float = 0.32
    online_learning_rate: float = 1e-4
    proximal_weight: float = 1e-3
    monitor_window: int = 30
    retrain_buffer: int = 200
    classification_calibration_mix: float = 0.5
    errors: deque[float] = field(default_factory=deque)

    def observe(self, predicted_class: int, true_class: int, cumulative_incidence: float, event: int) -> dict[str, float | str]:
        err = float(predicted_class != true_class) + self.classification_calibration_mix * abs(
            float(cumulative_incidence) - float(event)
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

    def soft_update_fusion_weights(
        self,
        weights: np.ndarray,
        gradient: np.ndarray,
        anchor_weights: np.ndarray,
    ) -> np.ndarray:
        """Apply the paper's proximal online step to fusion weights."""

        updated = weights - self.online_learning_rate * (
            gradient + self.proximal_weight * (weights - anchor_weights)
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
