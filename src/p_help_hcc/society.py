"""Phase A artificial HCC patient society projection."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    tiled = np.tile(block, (1, reps))
    return tiled[:, :width]


@dataclass
class SocietyTransformer:
    """Map 67 curated features into the paper's 46-dimensional agent state."""

    process_noise_std: float = 0.05
    scaler: StandardScaler | None = None

    def fit(self, x: np.ndarray) -> "SocietyTransformer":
        raw = self._project(x)
        self.scaler = StandardScaler().fit(raw)
        return self

    def transform(self, x: np.ndarray, *, add_noise: bool = False, seed: int | None = None) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("SocietyTransformer is not fitted")
        raw = self._project(x)
        state = self.scaler.transform(raw)
        if add_noise:
            rng = np.random.default_rng(seed)
            state = state + rng.normal(0.0, self.process_noise_std, size=state.shape)
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

    def _project(self, x: np.ndarray) -> np.ndarray:
        blocks = {name: x[:, start:end] for name, (start, end) in FEATURE_BLOCK_SLICES.items()}
        dem = _resize_block(blocks["dem"], 4)
        hep = _resize_block(blocks["hep"], 2)
        tum_summary = _resize_block(blocks["tum"], 3)
        liv_summary = _resize_block(blocks["lab"], 3)
        patient = np.hstack([dem, hep, tum_summary, liv_summary])

        tumor_base = _resize_block(blocks["tum"], 5)
        tumor = np.hstack([tumor_base, np.linalg.norm(tumor_base, axis=1, keepdims=True)])

        liver_base = _resize_block(blocks["lab"], 7)
        liver = np.hstack([liver_base, liver_base.mean(axis=1, keepdims=True)])

        treatment = _resize_block(blocks["trt"], SOCIETY_DIMS["treatment"])

        guideline_features = np.column_stack(
            [
                np.tanh(tumor[:, 0] - liver[:, 0]),
                np.tanh(liver[:, -1]),
                np.tanh(treatment[:, 1:].sum(axis=1)),
                np.ones(x.shape[0]),
            ]
        )

        explanation_features = np.column_stack(
            [
                _resize_block(blocks["fu"], 4),
                np.abs(tumor[:, :2]),
                np.abs(liver[:, :2]),
                treatment[:, :2],
            ]
        )
        explanation = _resize_block(explanation_features, SOCIETY_DIMS["explanation"])

        state = np.hstack([patient, tumor, liver, treatment, guideline_features, explanation])
        if state.shape[1] != AGENT_STATE_DIM:
            raise RuntimeError(f"Expected {AGENT_STATE_DIM} state dims, got {state.shape[1]}")
        return state.astype(float)


def force_action_in_state(state: np.ndarray, action_index: int) -> np.ndarray:
    out = state.copy()
    trt_slice = SocietyTransformer.block_indices("treatment")
    out[:, trt_slice] = 0.0
    out[:, trt_slice.start + action_index] = 1.0
    return out


@dataclass
class ArtificialSocietyDynamics:
    """Discrete Phase A rollout with tumor, liver, treatment, and guideline rules."""

    delta_t_months: float = 1.0
    horizon_months: int = 72
    d_max_cm: float = 20.0
    afp_alpha_per_month: float = 0.04
    afp_beta: float = 1.8
    fibrosis_age_rate_per_year: float = 0.002
    fibrosis_treatment_bump: float = 0.015
    fibrosis_recovery_rate: float = 0.020
    process_noise_std: float = 0.05
    action_effects: np.ndarray = field(
        default_factory=lambda: np.array([0.00, 0.18, 0.10, 0.14, 0.06, 0.16], dtype=float)
    )

    def treatment_policy(self, state: np.ndarray, guideline_mask: bool = True) -> np.ndarray:
        tumor = state[:, SocietyTransformer.block_indices("tumor")]
        liver = state[:, SocietyTransformer.block_indices("liver")]
        logits = np.column_stack(
            [
                -0.2 * tumor[:, 0] + 0.1 * liver[:, 0],
                -0.4 * tumor[:, 0] - 0.8 * liver[:, -1],
                0.2 * tumor[:, 0] - 0.2 * liver[:, -1],
                -0.1 * tumor[:, 0] - 0.3 * liver[:, -1],
                0.4 * tumor[:, 0] + 0.2 * liver[:, -1],
                0.2 * tumor[:, 0] - 0.1 * liver[:, -1],
            ]
        )
        if guideline_mask:
            infeasible_resection = liver[:, -1] > 1.25
            logits[infeasible_resection, 1] = -1e6
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)

    def step(self, state: np.ndarray, forced_action: int | None = None, seed: int | None = None) -> np.ndarray:
        out = state.copy()
        tumor_slice = SocietyTransformer.block_indices("tumor")
        liver_slice = SocietyTransformer.block_indices("liver")
        trt_slice = SocietyTransformer.block_indices("treatment")
        tumor = out[:, tumor_slice]
        liver = out[:, liver_slice]
        if forced_action is None:
            action = np.argmax(self.treatment_policy(out), axis=1)
        else:
            action = np.full(out.shape[0], forced_action, dtype=int)
        shrink = self.action_effects[action]
        diameter_proxy = np.clip(tumor[:, 0] + 2.5, 0.05, self.d_max_cm)
        growth = diameter_proxy * np.exp(0.03 * (1.0 - np.log(diameter_proxy) / np.log(self.d_max_cm)))
        tumor[:, 0] = np.clip(growth - shrink, 0.0, self.d_max_cm) - 2.5
        tumor[:, 2] = self.afp_alpha_per_month * tumor[:, 2] + self.afp_beta * np.log1p(np.clip(tumor[:, 0] + 2.5, 0, None))
        loco = np.isin(action, [1, 2, 3])
        recovery = np.clip(1.0 - 0.1 * np.maximum(liver[:, -1], 0.0), 0.0, 1.0)
        liver[:, -1] = liver[:, -1] + self.fibrosis_age_rate_per_year / 12.0
        liver[:, -1] += self.fibrosis_treatment_bump * loco
        liver[:, -1] -= self.fibrosis_recovery_rate * recovery
        out[:, tumor_slice] = tumor
        out[:, liver_slice] = liver
        out[:, trt_slice] = 0.0
        out[np.arange(out.shape[0]), trt_slice.start + action] = 1.0
        if self.process_noise_std > 0:
            rng = np.random.default_rng(seed)
            out = out + rng.normal(0.0, self.process_noise_std, size=out.shape)
        return out.astype(np.float32)

    def rollout(self, state: np.ndarray, forced_action: int | None = None, steps: int | None = None, seed: int | None = None) -> np.ndarray:
        steps = int(steps if steps is not None else self.horizon_months / self.delta_t_months)
        out = state.copy()
        for t in range(max(0, steps)):
            out = self.step(out, forced_action=forced_action, seed=None if seed is None else seed + t)
        return out
