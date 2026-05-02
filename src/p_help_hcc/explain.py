"""Explanation utilities: SHAP fallback and Cox-SHAP alignment."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Protocol

import numpy as np
from scipy.stats import spearmanr


class ProbabilityModel(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        ...


def permutation_attributions(
    model: ProbabilityModel,
    x: np.ndarray,
    *,
    repeats: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    base = model.predict_proba(x)
    pred_class = np.argmax(base, axis=1)
    base_score = base[np.arange(len(x)), pred_class]
    attr = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        drops = []
        for _ in range(repeats):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            pp = model.predict_proba(xp)
            drops.append(base_score - pp[np.arange(len(x)), pred_class])
        attr[:, j] = np.mean(drops, axis=0)
    return attr


def cox_shap_alignment(shap_values: np.ndarray, cox_beta: np.ndarray) -> dict[str, float]:
    shap_rank = np.argsort(np.mean(np.abs(shap_values), axis=0))
    cox_rank = np.argsort(np.abs(cox_beta))
    rho, p = spearmanr(shap_rank, cox_rank)
    return {"spearman_rho": float(rho), "spearman_p": float(p)}


def shap_available() -> bool:
    return find_spec("shap") is not None


def multinomial_brier_loss(proba: np.ndarray, y: np.ndarray) -> float:
    """Matured-only multinomial Brier score (used as the IPCW fallback)."""

    target = np.eye(proba.shape[1])[y]
    return float(np.mean(np.square(proba - target)))


def _km_censoring_estimator(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse Kaplan-Meier estimator of the censoring distribution G(t)."""

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    censor_event = (events == 0).astype(float)
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_censor = censor_event[order]
    unique_times = np.unique(sorted_times)
    surv = 1.0
    times_out = [0.0]
    surv_out = [1.0]
    for t in unique_times:
        at_risk = float((sorted_times >= t).sum())
        if at_risk <= 0:
            continue
        d_c = float(sorted_censor[sorted_times == t].sum())
        if d_c > 0:
            surv = surv * max(0.0, 1.0 - d_c / at_risk)
        times_out.append(float(t))
        surv_out.append(surv)
    return np.asarray(times_out), np.asarray(surv_out)


def _km_lookup(t: float, km_times: np.ndarray, km_surv: np.ndarray) -> float:
    if len(km_times) == 0:
        return 1.0
    idx = np.searchsorted(km_times, t, side="right") - 1
    idx = max(0, min(idx, len(km_surv) - 1))
    return float(km_surv[idx])


def ipcw_brier_loss(
    proba: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    cutpoints: list[float],
    *,
    floor: float = 1e-3,
) -> float:
    """IPCW-weighted multinomial Brier score per paper Eq.(13).

    Falls back to ``multinomial_brier_loss`` if censoring weights drop below
    the numerical floor on any fold.
    """

    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    n, n_classes = proba.shape
    if len(cutpoints) != n_classes - 1:
        return multinomial_brier_loss(proba, y)
    target = np.eye(n_classes)[y]
    sq = np.square(proba - target)
    km_times, km_surv = _km_censoring_estimator(times, events)
    total_weight = 0.0
    total_loss = 0.0
    for i in range(n):
        for c in range(n_classes):
            t_c = float(cutpoints[c - 1]) if c >= 1 else 0.0
            include = bool(events[i] == 1 or times[i] >= t_c)
            if not include:
                continue
            g = _km_lookup(min(float(times[i]), t_c) if c >= 1 else 0.0, km_times, km_surv)
            if g < floor:
                return multinomial_brier_loss(proba, y)
            w = 1.0 / g
            total_weight += w
            total_loss += w * float(sq[i, c])
    if total_weight <= 0:
        return multinomial_brier_loss(proba, y)
    return float(total_loss / total_weight)


def explanation_consistency_loss(
    shap_values: np.ndarray,
    cox_beta: np.ndarray,
    x: np.ndarray,
    *,
    kappa: float = 5.0,
) -> float:
    """L1 disagreement between SHAP signs and Cox interaction signs."""

    cox_target = np.tanh(kappa * (x * cox_beta.reshape(1, -1)))
    shap_target = np.tanh(kappa * shap_values)
    return float(np.mean(np.abs(shap_target - cox_target)))


def clinical_rule_violation_loss(shap_values: np.ndarray, feature_names: list[str]) -> float:
    """Simple knowledge-rule loss: known risk factors should not look protective globally."""

    risk_features = {"log_afp", "tumor_size_cm", "vascular_invasion", "extrahepatic_spread", "stage_iv"}
    penalties = []
    global_attr = np.mean(shap_values, axis=0)
    for idx, name in enumerate(feature_names):
        if name in risk_features:
            penalties.append(max(0.0, -float(global_attr[idx])))
    return float(np.mean(penalties)) if penalties else 0.0


def phase_e_loss_report(
    proba: np.ndarray,
    y: np.ndarray,
    shap_values: np.ndarray,
    cox_beta: np.ndarray,
    x: np.ndarray,
    feature_names: list[str],
    *,
    times: np.ndarray | None = None,
    events: np.ndarray | None = None,
    cutpoints: list[float] | None = None,
    lambda_cal: float = 1.0,
    lambda_exp: float = 0.2,
    lambda_clin: float = 0.1,
    kappa: float = 5.0,
) -> dict[str, float]:
    pred = float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))
    if times is not None and events is not None and cutpoints is not None:
        cal = ipcw_brier_loss(proba, y, times, events, cutpoints)
    else:
        cal = multinomial_brier_loss(proba, y)
    exp = explanation_consistency_loss(shap_values, cox_beta, x, kappa=kappa)
    clin = clinical_rule_violation_loss(shap_values, feature_names)
    total = pred + lambda_cal * cal + lambda_exp * exp + lambda_clin * clin
    return {
        "loss_pred": pred,
        "loss_cal": cal,
        "loss_exp": exp,
        "loss_clin": clin,
        "loss_total": float(total),
    }
