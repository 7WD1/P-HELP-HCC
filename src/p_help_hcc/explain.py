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
    target = np.eye(proba.shape[1])[y]
    return float(np.mean(np.square(proba - target)))


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
    lambda_cal: float = 1.0,
    lambda_exp: float = 0.2,
    lambda_clin: float = 0.1,
    kappa: float = 5.0,
) -> dict[str, float]:
    pred = float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))
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
