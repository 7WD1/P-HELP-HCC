"""Counterfactual treatment sweep for Phase C/E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import ACTION_SET
from .society import ArtificialSocietyDynamics, SocietyTransformer


class StateProbabilityModel(Protocol):
    def predict_proba_from_state(
        self, state: np.ndarray, cluster_one_hot: np.ndarray | None = None
    ) -> np.ndarray:
        ...


@dataclass
class PropensityModel:
    actions: list[str]
    model: LogisticRegression | None = None

    def fit(self, state: np.ndarray, observed_action: np.ndarray) -> "PropensityModel":
        y = np.asarray(observed_action, dtype=int)
        if len(np.unique(y)) < 2:
            self.model = None
        else:
            self.model = LogisticRegression(max_iter=1000).fit(state, y)
        return self

    def predict(self, state: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full((state.shape[0], len(self.actions)), 1.0 / len(self.actions))
        proba = self.model.predict_proba(state)
        full = np.zeros((state.shape[0], len(self.actions)), dtype=float)
        for idx, cls in enumerate(self.model.classes_):
            full[:, int(cls)] = proba[:, idx]
        missing = full.sum(axis=1) == 0
        full[missing] = 1.0 / len(self.actions)
        full = full / np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
        return full


def infer_observed_actions_from_state(state: np.ndarray, n_actions: int = 6) -> np.ndarray:
    trt = state[:, 26 : 26 + n_actions]
    return np.argmax(trt, axis=1).astype(int)


@dataclass
class CounterfactualSweep:
    actions: list[str] = None
    propensity_gate: tuple[float, float] = (0.05, 0.95)
    guideline_confidence_threshold: float = 0.6
    bootstrap_replicates: int = 200
    random_state: int = 42
    propensity_model: PropensityModel | None = None
    action_state_templates: np.ndarray | None = None
    dynamics: ArtificialSocietyDynamics | None = None

    def __post_init__(self) -> None:
        if self.actions is None:
            self.actions = list(ACTION_SET)

    def fit(self, state: np.ndarray, observed_actions: np.ndarray | None = None) -> "CounterfactualSweep":
        if observed_actions is None:
            observed_actions = infer_observed_actions_from_state(state, len(self.actions))
        self.propensity_model = PropensityModel(self.actions).fit(state, observed_actions)
        trt_slice = SocietyTransformer.block_indices("treatment")
        templates = []
        global_template = np.median(state[:, trt_slice], axis=0)
        for action_idx in range(len(self.actions)):
            mask = observed_actions == action_idx
            if np.any(mask):
                templates.append(np.median(state[mask][:, trt_slice], axis=0))
            else:
                templates.append(global_template)
        self.action_state_templates = np.vstack(templates)
        return self

    def sweep_patient(
        self,
        model: StateProbabilityModel,
        state_row: np.ndarray,
        *,
        clinical_only: bool = True,
        cluster_one_hot: np.ndarray | None = None,
    ) -> list[dict[str, float | str | bool]]:
        if self.propensity_model is None or self.action_state_templates is None:
            raise RuntimeError("CounterfactualSweep is not fitted")
        row = state_row.reshape(1, -1)
        prop = self.propensity_model.predict(row)[0]
        lo, hi = self.propensity_gate
        rng = np.random.default_rng(self.random_state)
        out: list[dict[str, float | str | bool]] = []
        trt_slice = SocietyTransformer.block_indices("treatment")
        distances = np.linalg.norm(self.action_state_templates - row[:, trt_slice], axis=1)
        factual_action = int(np.argmin(distances))
        factual_state = self._rollout_action(row, factual_action)
        factual_p = float(model.predict_proba_from_state(factual_state, cluster_one_hot)[:, 2:].sum(axis=1)[0])
        for action_idx, action in enumerate(self.actions):
            estimable = bool(lo <= prop[action_idx] <= hi)
            forced = self._rollout_action(row, action_idx)
            p = float(model.predict_proba_from_state(forced, cluster_one_hot)[:, 2:].sum(axis=1)[0])
            boot = []
            for _ in range(max(1, self.bootstrap_replicates)):
                noisy = forced + rng.normal(0.0, 0.03, size=forced.shape)
                boot.append(float(model.predict_proba_from_state(noisy, cluster_one_hot)[:, 2:].sum(axis=1)[0]))
            delta_boot = np.asarray(boot) - factual_p
            ci_lo, ci_hi = np.quantile(delta_boot, [0.025, 0.975])
            guideline_conf = float(np.clip(0.75 + 0.2 * estimable - 0.08 * (action_idx == 1 and row[0, 20] > 1.5), 0, 1))
            uncertain = bool(ci_lo <= 0.0 <= ci_hi)
            out.append(
                {
                    "action": action,
                    "propensity": float(prop[action_idx]),
                    "estimable": estimable,
                    "guideline_confidence": guideline_conf,
                    "passes_guideline": bool(guideline_conf >= self.guideline_confidence_threshold),
                    "p_os_gt_12m": p,
                    "delta_vs_factual": p - factual_p,
                    "delta_ci_low": float(ci_lo),
                    "delta_ci_high": float(ci_hi),
                    "uncertain": uncertain,
                }
            )
        ranked = sorted(out, key=lambda item: float(item["delta_vs_factual"]), reverse=True)
        if not clinical_only:
            return ranked
        return [
            item
            for item in ranked
            if bool(item["estimable"]) and bool(item["passes_guideline"]) and not bool(item["uncertain"])
        ]

    def _force_action(self, state: np.ndarray, action_index: int) -> np.ndarray:
        if self.action_state_templates is None:
            raise RuntimeError("CounterfactualSweep is not fitted")
        out = state.copy()
        trt_slice = SocietyTransformer.block_indices("treatment")
        out[:, trt_slice] = self.action_state_templates[action_index]
        return out

    def _rollout_action(self, state: np.ndarray, action_index: int) -> np.ndarray:
        forced = self._force_action(state, action_index)
        if self.dynamics is None:
            return forced
        return self.dynamics.rollout(forced, forced_action=action_index, steps=12, seed=self.random_state)


def extract_observed_actions_from_df(df) -> np.ndarray:
    """Map available treatment columns to the paper's six action indices."""

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    n = len(df)
    actions = np.zeros(n, dtype=int)
    if "surgical_strategy" in df.columns:
        strategy = df["surgical_strategy"].astype(str).str.lower()
        no_mask = strategy.str.contains("none|no", na=False)
        actions[no_mask.to_numpy()] = 0
        resection_mask = strategy.str.contains("resection", na=False) & ~no_mask
        actions[resection_mask.to_numpy()] = 1
        actions[strategy.str.contains("ablation|rfa", na=False).to_numpy()] = 3
    flag_order = [
        ("treatment_no_resection", 0),
        ("treatment_resection", 1),
        ("treatment_tace", 2),
        ("treatment_ablation", 3),
        ("treatment_rfa", 3),
        ("treatment_sorafenib", 4),
        ("treatment_combo", 5),
    ]
    for col, idx in flag_order:
        if col in df.columns:
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy() > 0
            actions[mask] = idx
    return actions.astype(int)
