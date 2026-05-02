"""Phase C heterogeneous learner ensemble and validation fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .constants import N_CLASSES
from .neural import MLPClassifierWrapper, train_mlp_classifier
from .utils import stable_softmax


def _complete_proba(model: Any, x: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = getattr(model, "classes_", np.arange(proba.shape[1]))
    out = np.zeros((x.shape[0], n_classes), dtype=float)
    for col, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, col]
    missing = out.sum(axis=1) <= 0
    out[missing] = 1.0 / n_classes
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of a vector onto the probability simplex."""

    if v.sum() == 1.0 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.full_like(v, 1.0 / len(v))
    rho_idx = rho[-1]
    theta = (cssv[rho_idx] - 1) / (rho_idx + 1)
    return np.maximum(v - theta, 0)


def optimize_brier_weights(proba_stack: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.25) -> np.ndarray:
    """Minimize multiclass Brier score over non-negative fusion weights."""

    n_models, n, n_classes = proba_stack.shape
    target = np.eye(n_classes)[y]
    w = np.full(n_models, 1.0 / n_models)
    for _ in range(steps):
        fused = np.tensordot(w, proba_stack, axes=(0, 0))
        grad = np.array([2.0 * np.mean((fused - target) * proba_stack[m]) for m in range(n_models)])
        w = project_simplex(w - lr * grad)
    return w


@dataclass
class CalibratedModel:
    base_model: Any
    calibrator: LogisticRegression | None = None

    def fit_calibrator(self, x_val: np.ndarray, y_val: np.ndarray, n_classes: int = N_CLASSES) -> "CalibratedModel":
        base_proba = _complete_proba(self.base_model, x_val, n_classes)
        if len(np.unique(y_val)) < 2:
            self.calibrator = None
        else:
            self.calibrator = LogisticRegression(max_iter=1000).fit(base_proba, y_val)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        base_proba = _complete_proba(self.base_model, x)
        if self.calibrator is None:
            return base_proba
        return _complete_proba(self.calibrator, base_proba)


@dataclass
class FusionCalibrationHead:
    alpha: float = 0.60
    model: LogisticRegression | None = None
    n_clusters: int = 4

    def fit(self, stacked_proba: np.ndarray, cluster_one_hot: np.ndarray, y: np.ndarray) -> "FusionCalibrationHead":
        features = np.hstack([stacked_proba, cluster_one_hot])
        if len(np.unique(y)) < 2:
            self.model = None
        else:
            self.model = LogisticRegression(max_iter=1000).fit(features, y)
        self.n_clusters = cluster_one_hot.shape[1]
        return self

    def predict(self, stacked_proba: np.ndarray, cluster_one_hot: np.ndarray) -> np.ndarray:
        if self.model is None:
            return stacked_proba
        features = np.hstack([stacked_proba, cluster_one_hot])
        calibrated = _complete_proba(self.model, features, stacked_proba.shape[1])
        return (1.0 - self.alpha) * stacked_proba + self.alpha * calibrated


@dataclass
class PHelpEnsemble:
    config: dict[str, Any]
    seed: int = 42
    models: dict[str, Any] = field(default_factory=dict)
    fusion_weights: np.ndarray | None = None
    calibration_head: FusionCalibrationHead | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        cluster_val: np.ndarray,
    ) -> "PHelpEnsemble":
        phase_c = self.config.get("phase_c", {})
        self.models["logistic"] = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=self.seed,
        ).fit(x_train, y_train)
        rf_cfg = phase_c.get("random_forest", {})
        rf_base = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 500)),
            max_depth=rf_cfg.get("max_depth", 10),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=self.seed,
            n_jobs=-1,
        ).fit(x_train, y_train)
        self.models["random_forest"] = CalibratedModel(rf_base).fit_calibrator(x_val, y_val)
        self.models["gradient_boosting_or_xgboost"] = self._fit_boosting(x_train, y_train)
        mlp_cfg = phase_c.get("mlp", {})
        self.models["mlp"] = train_mlp_classifier(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dims=list(mlp_cfg.get("hidden_dims", [256, 128, 64])),
            dropout=float(mlp_cfg.get("dropout", 0.2)),
            learning_rate=float(mlp_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(mlp_cfg.get("weight_decay", 0.0)),
            batch_size=int(mlp_cfg.get("batch_size", 32)),
            epochs=int(mlp_cfg.get("epochs", 100)),
            patience=int(mlp_cfg.get("patience", 15)),
            gamma=float(mlp_cfg.get("focal_gamma", 1.5)),
            class_weights=list(mlp_cfg.get("class_weights", [])) or None,
            seed=self.seed,
        )
        proba_stack = self.predict_base_stack(x_val)
        self.fusion_weights = optimize_brier_weights(proba_stack, y_val)
        stacked = self._fuse_stack(proba_stack)
        alpha = float(phase_c.get("fusion_alpha", 0.60))
        self.calibration_head = FusionCalibrationHead(alpha=alpha).fit(stacked, cluster_val, y_val)
        return self

    def _fit_boosting(self, x_train: np.ndarray, y_train: np.ndarray) -> Any:
        phase_c = self.config.get("phase_c", {})
        xgb_cfg = phase_c.get("xgboost", {})
        if xgb_cfg.get("enabled_if_installed", True) and find_spec("xgboost") is not None:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=int(xgb_cfg.get("n_estimators", 500)),
                learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                max_depth=int(xgb_cfg.get("max_depth", 6)),
                objective="multi:softprob",
                num_class=N_CLASSES,
                eval_metric="mlogloss",
                random_state=self.seed,
                n_jobs=-1,
            ).fit(x_train, y_train)
        gb_cfg = phase_c.get("gradient_boosting_fallback", {})
        return GradientBoostingClassifier(
            n_estimators=int(gb_cfg.get("n_estimators", 500)),
            learning_rate=float(gb_cfg.get("learning_rate", 0.05)),
            max_depth=int(gb_cfg.get("max_depth", 3)),
            random_state=self.seed,
        ).fit(x_train, y_train)

    def predict_base_stack(self, x: np.ndarray) -> np.ndarray:
        probs = []
        for name in ["logistic", "gradient_boosting_or_xgboost", "random_forest", "mlp"]:
            model = self.models[name]
            if isinstance(model, MLPClassifierWrapper):
                probs.append(model.predict_proba(x))
            else:
                probs.append(_complete_proba(model, x))
        return np.stack(probs, axis=0)

    def _fuse_stack(self, proba_stack: np.ndarray) -> np.ndarray:
        if self.fusion_weights is None:
            weights = np.full(proba_stack.shape[0], 1.0 / proba_stack.shape[0])
        else:
            weights = self.fusion_weights
        fused = np.tensordot(weights, proba_stack, axes=(0, 0))
        return fused / np.clip(fused.sum(axis=1, keepdims=True), 1e-12, None)

    def predict_proba(self, x: np.ndarray, cluster_one_hot: np.ndarray) -> np.ndarray:
        stacked = self._fuse_stack(self.predict_base_stack(x))
        if self.calibration_head is not None:
            stacked = self.calibration_head.predict(stacked, cluster_one_hot)
        return stable_softmax(np.log(np.clip(stacked, 1e-12, 1.0)), axis=1)
