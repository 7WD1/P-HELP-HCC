"""Contracts and calibration helpers for locked external validation.

Strict locked validation must not call ``fit`` on any target-cohort rows.  The
optional anchor calibrator in this module is deliberately a separate object: it
can only recalibrate already-frozen probabilities, and its outputs must be
reported as target-adapted sensitivity results rather than strict validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import N_CLASSES
from .ensemble import PhasePPlattCalibrator


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an immutable artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_contract(payload: dict[str, Any] | list[Any]) -> str:
    """Hash a canonical JSON contract, rejecting non-finite numeric state."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return np.asarray(value).tolist()


def _linear_calibrator_contract(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    contract: dict[str, Any] = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    for name in ("classes_", "coef_", "intercept_"):
        if hasattr(model, name):
            contract[name.rstrip("_")] = _array(getattr(model, name))
    return contract


def preprocessing_contract(model: Any) -> dict[str, Any]:
    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is None or not hasattr(preprocessor, "preprocessing_contract"):
        raise ValueError("Frozen model does not expose a fitted preprocessing contract")
    return preprocessor.preprocessing_contract()


def imputation_contract(model: Any) -> dict[str, Any]:
    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is None or not hasattr(preprocessor, "imputation_contract"):
        raise ValueError("Frozen model does not expose a fitted imputation contract")
    return preprocessor.imputation_contract()


def calibration_contract(model: Any) -> dict[str, Any]:
    """Describe every fitted calibration layer in the prediction path."""

    ensemble = getattr(model, "ensemble", None)
    if ensemble is None:
        raise ValueError("Frozen model does not expose an ensemble calibration path")
    base_calibrators: dict[str, Any] = {}
    for name, fitted in sorted(getattr(ensemble, "models", {}).items()):
        base_calibrators[str(name)] = _linear_calibrator_contract(
            getattr(fitted, "calibrator", None)
        )
    fusion = getattr(ensemble, "calibration_head", None)
    phase_p = getattr(ensemble, "phase_p_platt", None)
    return {
        "active_model_names": list(getattr(ensemble, "active_model_names_", [])),
        "fusion_weights": _array(getattr(ensemble, "fusion_weights", None)),
        "base_calibrators": base_calibrators,
        "fusion_calibration": None
        if fusion is None
        else {
            "alpha": float(fusion.alpha),
            "n_clusters": int(fusion.n_clusters),
            "model": _linear_calibrator_contract(fusion.model),
        },
        "phase_p_platt": None
        if phase_p is None
        else {
            "n_classes": int(phase_p.n_classes),
            "constants": _array(phase_p.constants),
            "models": [_linear_calibrator_contract(item) for item in phase_p.models],
        },
    }


def decision_contract(model: Any) -> dict[str, Any]:
    """Return the frozen prediction and monitoring decision rules.

    The eight-class classifier uses argmax, not target-cohort-fitted class
    thresholds.  Phase-P monitoring thresholds are included because they are
    also decisions carried by the serialized pipeline.
    """

    config = getattr(model, "config", {}) or {}
    phase_p = config.get("phase_p", {})
    return {
        "eight_class_rule": "argmax",
        "number_of_classes": int(N_CLASSES),
        "target_cohort_threshold_fitting": False,
        "phase_p_monitor": {
            "soft_error_threshold": float(phase_p.get("soft_error_threshold", 0.18)),
            "hard_error_threshold": float(phase_p.get("hard_error_threshold", 0.32)),
            "abstention_entropy_soft": float(phase_p.get("abstention_entropy_soft", 0.65)),
            "abstention_entropy_hard": float(phase_p.get("abstention_entropy_hard", 0.85)),
        },
    }


def model_contract_payloads(model: Any) -> dict[str, dict[str, Any]]:
    return {
        "preprocessing": preprocessing_contract(model),
        "imputation": imputation_contract(model),
        "calibration": calibration_contract(model),
        "decision": decision_contract(model),
    }


def model_contract_digests(model: Any) -> dict[str, str]:
    return {
        f"{name}_contract_sha256": hash_contract(payload)
        for name, payload in model_contract_payloads(model).items()
    }


def build_freeze_manifest(
    model: Any,
    model_path: str | Path,
    *,
    training_cohort: str = "Internal-673",
) -> dict[str, Any]:
    """Build the manifest that must be sealed before external-data access."""

    contracts = model_contract_payloads(model)
    return {
        "schema_version": 2,
        "training_cohort": str(training_cohort),
        "frozen_before_external_access": True,
        "model_sha256": sha256_file(model_path),
        "preprocessing_frozen": True,
        "imputation_frozen": True,
        "calibration_frozen": True,
        "thresholds_frozen": True,
        **{
            f"{name}_contract_sha256": hash_contract(payload)
            for name, payload in contracts.items()
        },
        "preprocessing_summary": {
            "curated_dimension": contracts["preprocessing"]["curated_dimension"],
            "missingness_sidecar": True,
        },
        "imputation_summary": {
            "strategy": contracts["imputation"]["strategy"],
            "fit_scope": contracts["imputation"]["fit_scope"],
        },
        "decision_summary": contracts["decision"],
    }


def verify_freeze_manifest(
    manifest: dict[str, Any], model: Any, model_path: str | Path
) -> dict[str, str]:
    """Verify the artifact and all prediction-path contracts before inference."""

    required = {
        "schema_version",
        "training_cohort",
        "frozen_before_external_access",
        "model_sha256",
        "preprocessing_frozen",
        "imputation_frozen",
        "calibration_frozen",
        "thresholds_frozen",
        "preprocessing_contract_sha256",
        "imputation_contract_sha256",
        "calibration_contract_sha256",
        "decision_contract_sha256",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Freeze manifest is missing fields: {missing}")
    if manifest["schema_version"] != 2:
        raise ValueError("Unsupported freeze-manifest schema_version; expected 2")
    if manifest["training_cohort"] != "Internal-673":
        raise ValueError("training_cohort must be Internal-673")
    for flag in (
        "frozen_before_external_access",
        "preprocessing_frozen",
        "imputation_frozen",
        "calibration_frozen",
        "thresholds_frozen",
    ):
        if manifest[flag] is not True:
            raise ValueError(f"{flag} must be true for locked validation")
    actual_model_hash = sha256_file(model_path)
    if str(manifest["model_sha256"]).lower() != actual_model_hash:
        raise ValueError("Frozen model hash does not match the freeze manifest")
    actual_contracts = model_contract_digests(model)
    for field, actual in actual_contracts.items():
        if str(manifest[field]).lower() != actual:
            raise ValueError(f"Frozen {field.removesuffix('_sha256')} does not match the model")
    return {"model_sha256": actual_model_hash, **actual_contracts}


@dataclass
class AnchorAffinePlattCalibrator:
    """Target-label affine/Platt sensitivity layer fitted on at most 30 anchors.

    This object never owns or mutates the frozen base model.  Each class receives
    an affine map on its frozen log-odds, followed by probability renormalization.
    """

    max_anchor_patients: int = 30
    random_state: int = 42
    calibrator: PhasePPlattCalibrator = field(
        default_factory=lambda: PhasePPlattCalibrator(n_classes=N_CLASSES)
    )
    n_anchor_patients_: int = 0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "AnchorAffinePlattCalibrator":
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
            raise ValueError(f"Anchor probabilities must have {N_CLASSES} columns")
        if len(probabilities) != len(labels) or len(labels) == 0:
            raise ValueError("Anchor probabilities and labels must be non-empty and aligned")
        if len(labels) > int(self.max_anchor_patients):
            raise ValueError(
                f"Anchor subset contains {len(labels)} patients; maximum is "
                f"{self.max_anchor_patients}"
            )
        if len(np.unique(labels)) < 2:
            raise ValueError("Anchor recalibration requires at least two observed classes")
        self.calibrator = PhasePPlattCalibrator(
            n_classes=N_CLASSES, random_state=self.random_state
        ).fit(probabilities, labels)
        self.n_anchor_patients_ = int(len(labels))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.n_anchor_patients_ <= 0:
            raise RuntimeError("AnchorAffinePlattCalibrator is not fitted")
        return self.calibrator.predict(np.asarray(probabilities, dtype=float))

    def contract(self) -> dict[str, Any]:
        if self.n_anchor_patients_ <= 0:
            raise RuntimeError("AnchorAffinePlattCalibrator is not fitted")
        return {
            "kind": "one-versus-rest affine Platt recalibration",
            "target_adaptation": True,
            "strict_locked_result": False,
            "n_anchor_patients": self.n_anchor_patients_,
            "max_anchor_patients": int(self.max_anchor_patients),
            "constants": _array(self.calibrator.constants),
            "models": [
                _linear_calibrator_contract(item) for item in self.calibrator.models
            ],
        }
