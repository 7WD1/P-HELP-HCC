"""Pretreatment-only observational scenario-sensitivity estimators.

This module provides cohort-level diagnostics and explicitly avoids presenting
model-based patient scenarios as individualized treatment-effect estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .statistics import e_value_from_risk_ratio


POST_ASSIGNMENT_TOKENS = (
    "treatment",
    "therapy",
    "resection",
    "ablation",
    "tace",
    "rfa",
    "sorafenib",
    "chemotherapy",
    "radiotherapy",
    "surgical_margin",
    "post_",
)


def binary_survival_outcome(
    times: Iterable[float], events: Iterable[int], horizon_months: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create an auditable horizon outcome and observed-outcome mask.

    Death on or before the horizon is coded 0 and follow-up beyond the horizon
    is coded 1.  Censoring before the horizon remains missing and is excluded.
    """

    time = np.asarray(list(times), dtype=float)
    event = np.asarray(list(events), dtype=int)
    if time.ndim != 1 or event.shape != time.shape:
        raise ValueError("times and events must be aligned one-dimensional arrays")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive")
    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("events must contain only 0/1 values")
    observed = ((event == 1) & (time <= horizon_months)) | (time >= horizon_months)
    outcome = (time > horizon_months).astype(float)
    return outcome, observed


def validate_pretreatment_feature_names(feature_names: Iterable[str]) -> list[str]:
    names = [str(name) for name in feature_names]
    flagged = [
        name
        for name in names
        if any(token in name.lower() for token in POST_ASSIGNMENT_TOKENS)
    ]
    if flagged:
        raise ValueError(
            "Propensity/outcome nuisance models must use pretreatment covariates only; "
            f"remove assignment-encoding or post-assignment fields: {flagged}"
        )
    return names


def _fit_binary_outcome(x: np.ndarray, y: np.ndarray, random_state: int):
    if len(y) == 0:
        return None, 0.5
    smoothed = float((y.sum() + 1.0) / (len(y) + 2.0))
    if len(np.unique(y)) < 2:
        return None, smoothed
    return LogisticRegression(max_iter=2000, random_state=random_state).fit(x, y), smoothed


def _predict_binary(model, fallback: float, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), fallback, dtype=float)
    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(x)[:, positive]


def _weighted_mean_and_var(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.clip(weight.sum(), 1e-12, None)
    mean = np.sum(x * weight[:, None], axis=0) / total
    var = np.sum(weight[:, None] * np.square(x - mean), axis=0) / total
    return mean, var


def standardized_mean_differences(
    x: np.ndarray,
    treatment: np.ndarray,
    action: int,
    reference: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    pair = np.isin(treatment, [action, reference])
    xa, xr = x[pair & (treatment == action)], x[pair & (treatment == reference)]
    if len(xa) == 0 or len(xr) == 0:
        raise ValueError("Both treatment arms need at least one observation")
    if weights is None:
        wa, wr = np.ones(len(xa)), np.ones(len(xr))
    else:
        wa = np.asarray(weights)[pair & (treatment == action)]
        wr = np.asarray(weights)[pair & (treatment == reference)]
    ma, va = _weighted_mean_and_var(xa, wa)
    mr, vr = _weighted_mean_and_var(xr, wr)
    return (ma - mr) / np.sqrt(np.clip((va + vr) / 2.0, 1e-12, None))


def cross_fitted_propensity_scores(
    pretreatment_covariates: np.ndarray,
    treatment: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold multinomial propensities and sorted action labels."""

    x = np.asarray(pretreatment_covariates, dtype=float)
    a = np.asarray(treatment, dtype=int)
    actions = np.sort(np.unique(a))
    counts = np.asarray([np.sum(a == action) for action in actions])
    folds = min(int(n_splits), int(counts.min()))
    if folds < 2:
        raise ValueError("Every treatment arm needs at least two rows for cross-fitting")
    action_to_col = {int(action): index for index, action in enumerate(actions)}
    propensity = np.zeros((len(x), len(actions)), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fold_id, (train_index, test_index) in enumerate(splitter.split(x, a)):
        model = LogisticRegression(max_iter=2000, random_state=random_state + fold_id).fit(
            x[train_index], a[train_index]
        )
        predicted = model.predict_proba(x[test_index])
        for model_column, action in enumerate(model.classes_):
            propensity[test_index, action_to_col[int(action)]] = predicted[:, model_column]
    propensity = np.clip(propensity, 1e-8, 1.0)
    propensity /= propensity.sum(axis=1, keepdims=True)
    return propensity, actions


def weighted_kaplan_meier_rmst(
    times: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    *,
    tau: float,
) -> float:
    """IPTW Kaplan--Meier restricted mean survival time through ``tau``."""

    time = np.asarray(times, dtype=float)
    event = np.asarray(events, dtype=int)
    weight = np.asarray(weights, dtype=float)
    if time.shape != event.shape or time.shape != weight.shape:
        raise ValueError("times, events, and weights must align")
    if tau <= 0 or np.any(weight < 0):
        raise ValueError("tau must be positive and weights non-negative")
    event_times = np.sort(np.unique(time[(event == 1) & (time <= tau)]))
    survival = 1.0
    previous = 0.0
    area = 0.0
    for event_time in event_times:
        area += survival * (float(event_time) - previous)
        at_risk_weight = float(weight[time >= event_time].sum())
        event_weight = float(weight[(time == event_time) & (event == 1)].sum())
        if at_risk_weight > 0:
            survival *= max(0.0, 1.0 - event_weight / at_risk_weight)
        previous = float(event_time)
    area += survival * max(0.0, float(tau) - previous)
    return float(area)


def iptw_rmst_contrasts(
    pretreatment_covariates: np.ndarray,
    treatment: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    *,
    reference_action: int,
    tau: float,
    trim: float = 0.05,
    n_splits: int = 5,
    bootstrap_replicates: int = 1000,
    random_state: int = 42,
) -> list[dict[str, object]]:
    """Cross-fitted IPTW-KM RMST contrasts with patient bootstrap refitting."""

    x = np.asarray(pretreatment_covariates, dtype=float)
    a = np.asarray(treatment, dtype=int)
    time = np.asarray(times, dtype=float)
    event = np.asarray(events, dtype=int)
    propensity, actions = cross_fitted_propensity_scores(
        x, a, n_splits=n_splits, random_state=random_state
    )
    action_to_col = {int(action): index for index, action in enumerate(actions)}
    if reference_action not in action_to_col:
        raise ValueError("reference_action is absent")

    def _contrast(
        x_local: np.ndarray,
        a_local: np.ndarray,
        time_local: np.ndarray,
        event_local: np.ndarray,
        action: int,
        seed: int,
        *,
        use_crossfit: bool,
    ) -> tuple[float, int, int]:
        present = set(map(int, np.unique(a_local)))
        if action not in present or int(reference_action) not in present or len(present) < 2:
            return float("nan"), 0, len(a_local)
        if use_crossfit:
            local_propensity, local_actions = cross_fitted_propensity_scores(
                x_local, a_local, n_splits=n_splits, random_state=seed
            )
        else:
            fitted = LogisticRegression(max_iter=2000, random_state=seed).fit(x_local, a_local)
            local_actions = np.asarray(fitted.classes_, dtype=int)
            local_propensity = fitted.predict_proba(x_local)
        local_map = {int(label): index for index, label in enumerate(local_actions)}
        if action not in local_map or reference_action not in local_map:
            return float("nan"), 0, len(a_local)
        action_col, ref_col = local_map[action], local_map[reference_action]
        support = (
            (local_propensity[:, action_col] >= trim)
            & (local_propensity[:, ref_col] >= trim)
            & (local_propensity[:, action_col] <= 1.0 - trim)
            & (local_propensity[:, ref_col] <= 1.0 - trim)
        )
        action_rows = support & (a_local == action)
        ref_rows = support & (a_local == reference_action)
        if not action_rows.any() or not ref_rows.any():
            return float("nan"), int(support.sum()), len(a_local)
        action_rmst = weighted_kaplan_meier_rmst(
            time_local[action_rows],
            event_local[action_rows],
            1.0 / local_propensity[action_rows, action_col],
            tau=tau,
        )
        ref_rmst = weighted_kaplan_meier_rmst(
            time_local[ref_rows],
            event_local[ref_rows],
            1.0 / local_propensity[ref_rows, ref_col],
            tau=tau,
        )
        return action_rmst - ref_rmst, int(support.sum()), len(a_local)

    rng = np.random.default_rng(random_state)
    results: list[dict[str, object]] = []
    for action in actions:
        action = int(action)
        if action == int(reference_action):
            continue
        estimate, retained, total = _contrast(
            x, a, time, event, action, random_state, use_crossfit=True
        )
        draws = []
        for bootstrap_index in range(bootstrap_replicates):
            sampled = rng.integers(0, len(x), size=len(x))
            draw, _retained, _total = _contrast(
                x[sampled],
                a[sampled],
                time[sampled],
                event[sampled],
                action,
                random_state + bootstrap_index + 1,
                use_crossfit=False,
            )
            if np.isfinite(draw):
                draws.append(draw)
        low, high = (
            np.quantile(draws, [0.025, 0.975])
            if draws
            else (float("nan"), float("nan"))
        )
        results.append(
            {
                "action": action,
                "reference_action": int(reference_action),
                "tau_months": float(tau),
                "iptw_km_rmst_difference_months": float(estimate),
                "ci_low": float(low),
                "ci_high": float(high),
                "retained": int(retained),
                "total": int(total),
                "retained_fraction": float(retained / max(1, total)),
                "bootstrap_unit": "patient",
                "bootstrap_refits_propensity_model": True,
                "bootstrap_replicates_requested": int(bootstrap_replicates),
                "bootstrap_replicates_valid": int(len(draws)),
            }
        )
    return results


@dataclass
class CrossFittedDoublyRobust:
    """Cross-fitted AIPW analysis with patient-level score bootstraps."""

    n_splits: int = 5
    trim: float = 0.05
    bootstrap_replicates: int = 1000
    random_state: int = 42
    action_labels: dict[int, str] = field(default_factory=dict)
    nuisance_propensity_: np.ndarray | None = None
    nuisance_outcome_: np.ndarray | None = None
    actions_: np.ndarray | None = None
    results_: dict[str, object] = field(default_factory=dict)

    def fit(
        self,
        pretreatment_covariates: np.ndarray,
        treatment: Iterable[int],
        outcome: Iterable[float],
        *,
        feature_names: Iterable[str],
        reference_action: int,
    ) -> "CrossFittedDoublyRobust":
        names = validate_pretreatment_feature_names(feature_names)
        x = np.asarray(pretreatment_covariates, dtype=float)
        a = np.asarray(list(treatment), dtype=int)
        y = np.asarray(list(outcome), dtype=float)
        if x.ndim != 2 or x.shape[0] != len(a) or len(a) != len(y):
            raise ValueError("covariates, treatment, and outcome must have aligned rows")
        if x.shape[1] != len(names):
            raise ValueError("feature_names must align with pretreatment_covariates columns")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("nuisance-model inputs cannot contain missing/non-finite values")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("outcome must be a binary observed horizon outcome")
        actions = np.sort(np.unique(a))
        if reference_action not in actions:
            raise ValueError("reference_action is absent from treatment")
        arm_counts = np.array([np.sum(a == action) for action in actions])
        folds = min(int(self.n_splits), int(arm_counts.min()))
        if folds < 2:
            raise ValueError("Every treatment arm must have at least two patients for cross-fitting")
        if not 0 <= self.trim < 0.5:
            raise ValueError("trim must lie in [0, 0.5)")

        action_to_col = {int(action): idx for idx, action in enumerate(actions)}
        propensity = np.zeros((len(x), len(actions)), dtype=float)
        outcome_regression = np.zeros_like(propensity)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(x, a)):
            propensity_model = LogisticRegression(
                max_iter=2000,
                random_state=self.random_state + fold_id,
            ).fit(x[train_idx], a[train_idx])
            fold_propensity = propensity_model.predict_proba(x[test_idx])
            for model_col, action in enumerate(propensity_model.classes_):
                propensity[test_idx, action_to_col[int(action)]] = fold_propensity[:, model_col]
            for action in actions:
                arm_train = train_idx[a[train_idx] == action]
                model, fallback = _fit_binary_outcome(
                    x[arm_train], y[arm_train], self.random_state + fold_id + int(action)
                )
                outcome_regression[test_idx, action_to_col[int(action)]] = _predict_binary(
                    model, fallback, x[test_idx]
                )

        propensity = np.clip(propensity, 1e-8, 1.0)
        propensity /= propensity.sum(axis=1, keepdims=True)
        self.nuisance_propensity_ = propensity
        self.nuisance_outcome_ = outcome_regression
        self.actions_ = actions

        rng = np.random.default_rng(self.random_state)
        contrasts: list[dict[str, object]] = []
        ref_col = action_to_col[int(reference_action)]
        for action in actions:
            if int(action) == int(reference_action):
                continue
            action_col = action_to_col[int(action)]
            support = (
                (propensity[:, action_col] >= self.trim)
                & (propensity[:, ref_col] >= self.trim)
                & (propensity[:, action_col] <= 1.0 - self.trim)
                & (propensity[:, ref_col] <= 1.0 - self.trim)
            )
            if not support.any():
                raise ValueError(f"No overlap remains for action {action} versus {reference_action}")
            psi_action = outcome_regression[:, action_col] + (a == action) * (
                y - outcome_regression[:, action_col]
            ) / propensity[:, action_col]
            psi_ref = outcome_regression[:, ref_col] + (a == reference_action) * (
                y - outcome_regression[:, ref_col]
            ) / propensity[:, ref_col]
            iptw = np.zeros(len(x), dtype=float)
            iptw[a == action] = 1.0 / propensity[a == action, action_col]
            iptw[a == reference_action] = 1.0 / propensity[a == reference_action, ref_col]
            iptw[~support] = 0.0
            before = standardized_mean_differences(x[support], a[support], int(action), int(reference_action))
            after = standardized_mean_differences(
                x[support], a[support], int(action), int(reference_action), iptw[support]
            )
            support_index = np.flatnonzero(support)

            def _three_estimators(index: np.ndarray) -> tuple[float, float, float, float]:
                assigned = a[index]
                observed = y[index]
                action_mask = assigned == action
                ref_mask = assigned == reference_action
                if not action_mask.any() or not ref_mask.any():
                    return (float("nan"),) * 4
                naive = float(observed[action_mask].mean() - observed[ref_mask].mean())
                action_weight = 1.0 / propensity[index[action_mask], action_col]
                ref_weight = 1.0 / propensity[index[ref_mask], ref_col]
                mu_action_iptw = float(np.sum(action_weight * observed[action_mask]) / np.sum(action_weight))
                mu_ref_iptw = float(np.sum(ref_weight * observed[ref_mask]) / np.sum(ref_weight))
                iptw_difference = mu_action_iptw - mu_ref_iptw
                mu_action_dr = float(np.mean(psi_action[index]))
                mu_ref_dr = float(np.mean(psi_ref[index]))
                dr_difference = mu_action_dr - mu_ref_dr
                dr_risk_ratio = float(
                    np.clip(mu_action_dr, 1e-6, 1.0)
                    / np.clip(mu_ref_dr, 1e-6, 1.0)
                )
                return naive, iptw_difference, dr_difference, dr_risk_ratio

            naive_estimate, iptw_estimate, dr_estimate, dr_rr = _three_estimators(support_index)
            bootstrap_draws = np.empty((self.bootstrap_replicates, 4), dtype=float)
            for bootstrap_index in range(self.bootstrap_replicates):
                sampled = support_index[
                    rng.integers(0, len(support_index), size=len(support_index))
                ]
                bootstrap_draws[bootstrap_index] = _three_estimators(sampled)

            def _estimate_summary(estimate: float, column: int) -> dict[str, float | int | str]:
                finite = bootstrap_draws[:, column][np.isfinite(bootstrap_draws[:, column])]
                if len(finite) == 0:
                    low = high = float("nan")
                else:
                    low, high = np.quantile(finite, [0.025, 0.975])
                return {
                    "risk_difference": float(estimate),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_replicates_requested": int(self.bootstrap_replicates),
                    "bootstrap_replicates_valid": int(len(finite)),
                    "bootstrap_unit": "patient",
                }

            naive_summary = _estimate_summary(naive_estimate, 0)
            iptw_summary = _estimate_summary(iptw_estimate, 1)
            dr_summary = _estimate_summary(dr_estimate, 2)
            finite_rr = bootstrap_draws[:, 3][np.isfinite(bootstrap_draws[:, 3])]
            rr_low, rr_high = (
                np.quantile(finite_rr, [0.025, 0.975])
                if len(finite_rr)
                else (float("nan"), float("nan"))
            )
            if np.isfinite(rr_low) and np.isfinite(rr_high):
                closest_bound = rr_low if dr_rr >= 1.0 else rr_high
                if rr_low <= 1.0 <= rr_high:
                    closest_bound = 1.0
                e_value_bound = e_value_from_risk_ratio(float(closest_bound))
            else:
                e_value_bound = float("nan")
            retention_by_arm = {
                str(int(arm)): {
                    "retained": int(np.sum(support & (a == arm))),
                    "total": int(np.sum(a == arm)),
                    "fraction": float(
                        np.sum(support & (a == arm)) / max(1, np.sum(a == arm))
                    ),
                }
                for arm in (action, reference_action)
            }
            contrasts.append(
                {
                    "action": int(action),
                    "action_label": self.action_labels.get(int(action), str(int(action))),
                    "reference_action": int(reference_action),
                    "reference_label": self.action_labels.get(int(reference_action), str(int(reference_action))),
                    "estimators": {
                        "naive": naive_summary,
                        "iptw": iptw_summary,
                        "aipw_dr": dr_summary,
                    },
                    "risk_difference": float(dr_estimate),
                    "ci_low": float(dr_summary["ci_low"]),
                    "ci_high": float(dr_summary["ci_high"]),
                    "dr_risk_ratio": float(dr_rr),
                    "dr_risk_ratio_ci_low": float(rr_low),
                    "dr_risk_ratio_ci_high": float(rr_high),
                    "e_value_point": e_value_from_risk_ratio(dr_rr),
                    "e_value_ci_bound": float(e_value_bound),
                    "on_support_n": int(support.sum()),
                    "on_support_fraction": float(support.mean()),
                    "retention_by_observed_arm": retention_by_arm,
                    "max_abs_smd_before": float(np.max(np.abs(before))),
                    "max_abs_smd_after_iptw": float(np.max(np.abs(after))),
                    "bootstrap_replicates": int(self.bootstrap_replicates),
                }
            )
        self.results_ = {
            "analysis": "observational_scenario_sensitivity",
            "causal_claim": False,
            "pretreatment_feature_names": names,
            "crossfit_folds": folds,
            "propensity_trim": self.trim,
            "bootstrap_unit": "patient",
            "bootstrap_refits_nuisance_models": False,
            "bootstrap_note": "patient rows are resampled; cross-fitted nuisance predictions remain fixed",
            "contrasts": contrasts,
            "arm_counts": {str(int(action)): int(np.sum(a == action)) for action in actions},
        }
        return self

    def report(self) -> dict[str, object]:
        if not self.results_:
            raise RuntimeError("CrossFittedDoublyRobust is not fitted")
        return self.results_
