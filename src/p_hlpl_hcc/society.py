"""Static six-block patient-state projection for Phase A.

The projected state is standardized and therefore has no centimeter/month
interpretation. Physical-unit longitudinal equations are intentionally kept
out of this module and out of the classifier prediction path; they are
available only through ``scripts/run_dynamics_backtest.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .constants import AGENT_STATE_DIM, FEATURE_BLOCK_SLICES, SOCIETY_DIMS


def _resize_block(block: np.ndarray, width: int) -> np.ndarray:
    if block.shape[1] == width:
        return block
    if block.shape[1] == 0:
        return np.zeros((block.shape[0], width), dtype=float)
    if block.shape[1] > width:
        chunks = np.array_split(block, width, axis=1)
        return np.column_stack([chunk.mean(axis=1) for chunk in chunks])
    reps = int(np.ceil(width / block.shape[1]))
    return np.tile(block, (1, reps))[:, :width]


class SocietyTransformer:
    """Map 67 curated features into a standardized 46-dimensional state."""

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None

    def fit(self, x: np.ndarray) -> "SocietyTransformer":
        self.scaler = StandardScaler().fit(self._project(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("SocietyTransformer is not fitted")
        state = self.scaler.transform(self._project(x))
        return state.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    @staticmethod
    def block_indices(name: str) -> slice:
        offset = 0
        for block_name, width in SOCIETY_DIMS.items():
            start = offset
            offset += width
            if block_name == name:
                return slice(start, offset)
        raise KeyError(name)

    @staticmethod
    def action_derived_auxiliary_indices() -> tuple[int, int, int]:
        """Coordinates that directly copy/summarize the factual treatment block."""

        guideline = SocietyTransformer.block_indices("guideline")
        explanation = SocietyTransformer.block_indices("explanation")
        return (
            guideline.start + 2,
            explanation.stop - 2,
            explanation.stop - 1,
        )

    def _project(self, x: np.ndarray) -> np.ndarray:
        blocks = {
            name: x[:, start:end]
            for name, (start, end) in FEATURE_BLOCK_SLICES.items()
        }
        dem = _resize_block(blocks["dem"], 4)
        hep = _resize_block(blocks["hep"], 2)
        tumor_summary = _resize_block(blocks["tum"], 3)
        liver_summary = _resize_block(blocks["lab"], 3)
        patient = np.hstack([dem, hep, tumor_summary, liver_summary])

        tumor_base = _resize_block(blocks["tum"], 5)
        tumor = np.hstack(
            [tumor_base, np.linalg.norm(tumor_base, axis=1, keepdims=True)]
        )

        liver_base = _resize_block(blocks["lab"], 7)
        liver = np.hstack([liver_base, liver_base.mean(axis=1, keepdims=True)])
        treatment = _resize_block(blocks["trt"], SOCIETY_DIMS["treatment"])

        guideline = np.column_stack(
            [
                np.tanh(tumor[:, 0] - liver[:, 0]),
                np.tanh(liver[:, -1]),
                np.tanh(treatment[:, 1:].sum(axis=1)),
                np.ones(x.shape[0]),
            ]
        )

        explanation_source = np.column_stack(
            [
                _resize_block(blocks["fu"], 4),
                np.abs(tumor[:, :2]),
                np.abs(liver[:, :2]),
                treatment[:, :2],
            ]
        )
        explanation = _resize_block(
            explanation_source, SOCIETY_DIMS["explanation"]
        )

        state = np.hstack(
            [patient, tumor, liver, treatment, guideline, explanation]
        )
        if state.shape[1] != AGENT_STATE_DIM:
            raise RuntimeError(
                f"Expected {AGENT_STATE_DIM} state dims, got {state.shape[1]}"
            )
        return state.astype(float)
