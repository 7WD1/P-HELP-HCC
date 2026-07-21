"""Model-based observational treatment-scenario sweep for Phase C/E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import ACTION_SET
from .society import SocietyTransformer


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
    guideline_confidence_threshold: float = 0.30
    bootstrap_replicates: int = 200
    random_state: int = 42
    propensity_model: PropensityModel | None = None
    action_state_templates: np.ndarray | None = None
    propensity_feature_count_: int = 0

    def __post_init__(self) -> None:
        if self.actions is None:
            self.actions = list(ACTION_SET)

    def fit(self, state: np.ndarray, observed_actions: np.ndarray | None = None) -> "CounterfactualSweep":
        if observed_actions is None:
            observed_actions = infer_observed_actions_from_state(state, len(self.actions))
        trt_slice = SocietyTransformer.block_indices("treatment")
        # Only Patient/Tumor/Liver blocks precede the Treatment block.  Later
        # Guideline and Explanation blocks also encode treatment information,
        # so neither they nor the Treatment block enter the propensity model.
        pretreatment_state = np.asarray(state[:, : trt_slice.start], dtype=float)
        self.propensity_feature_count_ = int(pretreatment_state.shape[1])
        self.propensity_model = PropensityModel(self.actions).fit(pretreatment_state, observed_actions)
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
        observed_action: int | None = None,
        clinical_only: bool = True,
        cluster_one_hot: np.ndarray | None = None,
        patient_bootstrap_predictions: dict[int, np.ndarray] | None = None,
        guideline_confidence_by_action: dict[int, float] | None = None,
    ) -> list[dict[str, object]]:
        if self.propensity_model is None or self.action_state_templates is None:
            raise RuntimeError("CounterfactualSweep is not fitted")
        if observed_action is None:
            raise ValueError(
                "A recorded pretreatment action is required as the factual arm; "
                "it is never inferred from proximity to a treatment template"
            )
        factual_action = int(observed_action)
        if factual_action < 0 or factual_action >= len(self.actions):
            raise ValueError(
                f"observed_action must be an integer in 0..{len(self.actions) - 1}"
            )
        row = state_row.reshape(1, -1)
        prop = self.propensity_model.predict(row[:, : self.propensity_feature_count_])[0]
        lo, hi = self.propensity_gate
        out: list[dict[str, object]] = []
        factual_state = self._rollout_action(row, factual_action)
        factual_p = float(model.predict_proba_from_state(factual_state, cluster_one_hot)[:, 2:].sum(axis=1)[0])
        for action_idx, action in enumerate(self.actions):
            estimable = bool(lo <= prop[action_idx] <= hi)
            forced = self._rollout_action(row, action_idx)
            p = float(model.predict_proba_from_state(forced, cluster_one_hot)[:, 2:].sum(axis=1)[0])
            bootstrap_available = bool(
                patient_bootstrap_predictions is not None
                and action_idx in patient_bootstrap_predictions
                and factual_action in patient_bootstrap_predictions
            )
            if bootstrap_available:
                action_draws = np.asarray(patient_bootstrap_predictions[action_idx], dtype=float)
                factual_draws = np.asarray(patient_bootstrap_predictions[factual_action], dtype=float)
                if action_draws.shape != factual_draws.shape or action_draws.ndim != 1:
                    raise ValueError("patient bootstrap prediction arrays must be aligned vectors")
                if len(action_draws) != self.bootstrap_replicates:
                    raise ValueError(
                        "patient bootstrap prediction arrays must contain exactly "
                        f"B={self.bootstrap_replicates} draws"
                    )
                if not np.all(np.isfinite(action_draws)) or not np.all(
                    np.isfinite(factual_draws)
                ):
                    raise ValueError("patient bootstrap prediction arrays must be finite")
                delta_draws = action_draws - factual_draws
                ci_lo, ci_hi = np.quantile(delta_draws, [0.025, 0.975])
                uncertain = bool(ci_lo <= 0.0 <= ci_hi)
            else:
                ci_lo = ci_hi = None
                uncertain = True
            guideline_available = bool(
                guideline_confidence_by_action is not None
                and action_idx in guideline_confidence_by_action
            )
            guideline_conf = (
                float(guideline_confidence_by_action[action_idx])
                if guideline_available
                else float("nan")
            )
            passes_guideline = bool(
                guideline_available
                and guideline_conf >= self.guideline_confidence_threshold
            )
            out.append(
                {
                    "action": action,
                    "action_index": int(action_idx),
                    "factual_action": self.actions[factual_action],
                    "factual_action_index": factual_action,
                    "factual_action_source": "recorded_pretreatment_action",
                    "propensity": float(prop[action_idx]),
                    "estimable": estimable,
                    "guideline_confidence": guideline_conf,
                    "passes_guideline": passes_guideline,
                    "p_os_gt_12m": p,
                    "delta_vs_factual": p - factual_p,
                    "delta_ci_low": None if ci_lo is None else float(ci_lo),
                    "delta_ci_high": None if ci_hi is None else float(ci_hi),
                    "uncertainty_available": bootstrap_available,
                    "bootstrap_replicates_used": int(len(action_draws))
                    if bootstrap_available
                    else 0,
                    "uncertainty_method": "patient_level_bootstrap"
                    if bootstrap_available
                    else "requires_patient_level_bootstrap",
                    "uncertain": uncertain,
                    "interpretation": "model_based_observational_scenario_only",
                }
            )
        ranked = sorted(out, key=lambda item: float(item["delta_vs_factual"]), reverse=True)
        if not clinical_only:
            return ranked
        return [
            item
            for item in ranked
            if bool(item["estimable"])
            and bool(item["passes_guideline"])
            and bool(item["uncertainty_available"])
            and not bool(item["uncertain"])
        ]

    def _force_action(self, state: np.ndarray, action_index: int) -> np.ndarray:
        if self.action_state_templates is None:
            raise RuntimeError("CounterfactualSweep is not fitted")
        out = state.copy()
        trt_slice = SocietyTransformer.block_indices("treatment")
        out[:, trt_slice] = self.action_state_templates[action_index]
        # The projected Guideline and Explanation blocks contain three
        # coordinates derived directly from the factual Treatment block. The
        # scaler prevents an exact nonlinear recomputation from standardized
        # state alone, so every arm (including the factual baseline) receives
        # the same neutral value on those coordinates. This prevents factual
        # treatment leakage into the main survival sweep.
        out[:, SocietyTransformer.action_derived_auxiliary_indices()] = 0.0
        return out

    def _rollout_action(self, state: np.ndarray, action_index: int) -> np.ndarray:
        # Main-path state is standardized, so physical-unit dynamics are never
        # applied here. Longitudinal equations live in the separate back-test.
        return self._force_action(state, action_index)


def extract_observed_actions_from_df(
    df, *, require_explicit: bool = False
) -> np.ndarray:
    """Map recorded pretreatment fields to the paper's six action indices.

    When ``require_explicit`` is true, every row must contain a recognized
    recorded action.  This is required for a patient-level factual contrast;
    the factual arm is never reconstructed from state-template proximity.
    """

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    n = len(df)
    actions = np.zeros(n, dtype=int)
    recognized = np.zeros(n, dtype=bool)
    if "surgical_strategy" in df.columns:
        strategy = df["surgical_strategy"].astype(str).str.lower()
        no_mask = strategy.str.contains(
            r"^(?:none|no(?:\s+resection|\s+surgery)?|best\s+supportive\s+care|bsc|palliative)$",
            regex=True,
            na=False,
        )
        actions[no_mask.to_numpy()] = 0
        recognized |= no_mask.to_numpy()
        resection_mask = strategy.str.contains("resection", na=False) & ~no_mask
        actions[resection_mask.to_numpy()] = 1
        recognized |= resection_mask.to_numpy()
        tace_mask = strategy.str.contains("tace|chemoembol", na=False)
        actions[tace_mask.to_numpy()] = 2
        recognized |= tace_mask.to_numpy()
        ablation_mask = strategy.str.contains("ablation|rfa", na=False)
        actions[ablation_mask.to_numpy()] = 3
        recognized |= ablation_mask.to_numpy()
        systemic_mask = strategy.str.contains("sorafenib|systemic", na=False)
        actions[systemic_mask.to_numpy()] = 4
        recognized |= systemic_mask.to_numpy()
        combo_mask = strategy.str.contains("combo|combination", na=False)
        actions[combo_mask.to_numpy()] = 5
        recognized |= combo_mask.to_numpy()
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
            recognized |= mask
    if require_explicit and not np.all(recognized):
        missing_rows = np.flatnonzero(~recognized)
        preview = missing_rows[:5].tolist()
        raise ValueError(
            "A recorded pretreatment action is required for every requested "
            f"factual contrast; unrecognized rows include {preview}"
        )
    return actions.astype(int)
