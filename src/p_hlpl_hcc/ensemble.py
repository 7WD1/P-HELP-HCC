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
from .parallel import phase_p_ipcw_residual_weights
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
class PhasePPlattCalibrator:
    """One-vs-rest Platt scaling fitted on the inner validation stream."""

    n_classes: int = N_CLASSES
    random_state: int = 42
    models: list[LogisticRegression | None] = field(default_factory=list)
    constants: np.ndarray | None = None

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "PhasePPlattCalibrator":
        probabilities = np.asarray(probabilities, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes = probabilities.shape[1]
        self.models = []
        constants = []
        for class_index in range(self.n_classes):
            binary = (y == class_index).astype(int)
            # Laplace smoothing also provides a deterministic fallback when a
            # small validation split does not contain both binary outcomes.
            constants.append(float((binary.sum() + 1.0) / (len(binary) + 2.0)))
            if len(np.unique(binary)) < 2:
                self.models.append(None)
                continue
            score = self._logit(probabilities[:, class_index]).reshape(-1, 1)
            self.models.append(
                LogisticRegression(max_iter=1000, random_state=self.random_state).fit(score, binary)
            )
        self.constants = np.asarray(constants, dtype=float)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.constants is None or len(self.models) != probabilities.shape[1]:
            raise RuntimeError("PhasePPlattCalibrator is not fitted")
        probabilities = np.asarray(probabilities, dtype=float)
        calibrated = np.zeros_like(probabilities, dtype=float)
        for class_index, model in enumerate(self.models):
            if model is None:
                calibrated[:, class_index] = self.constants[class_index]
                continue
            score = self._logit(probabilities[:, class_index]).reshape(-1, 1)
            positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
            calibrated[:, class_index] = model.predict_proba(score)[:, positive_column]
        return calibrated / np.clip(calibrated.sum(axis=1, keepdims=True), 1e-12, None)


@dataclass
class PHlplEnsemble:
    config: dict[str, Any]
    seed: int = 42
    models: dict[str, Any] = field(default_factory=dict)
    fusion_weights: np.ndarray | None = None
    calibration_head: FusionCalibrationHead | None = None
    phase_p_platt: PhasePPlattCalibrator | None = None
    phase_p_sample_weights_: np.ndarray | None = None
    mechanism_trace_: dict[str, Any] = field(default_factory=dict)
    active_model_names_: list[str] = field(default_factory=list)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        cluster_val: np.ndarray,
        *,
        scenario_train: np.ndarray | None = None,
        scenario_val: np.ndarray | None = None,
        train_times: np.ndarray | None = None,
        train_events: np.ndarray | None = None,
        val_times: np.ndarray | None = None,
        val_events: np.ndarray | None = None,
        cutpoints: list[float] | None = None,
        cox_direction: np.ndarray | None = None,
    ) -> "PHlplEnsemble":
        phase_c = self.config.get("phase_c", {})
        requested = [str(name).lower() for name in phase_c.get("learners", [])]
        aliases = {
            "logistic": "logistic",
            "gradient_boosting_or_xgboost": "gradient_boosting_or_xgboost",
            "gradient_boosting": "gradient_boosting_or_xgboost",
            "xgboost": "gradient_boosting_or_xgboost",
            "random_forest": "random_forest",
            "mlp": "mlp",
            "deep_encoder": "mlp",
        }
        self.active_model_names_ = []
        for name in requested or list(aliases):
            canonical = aliases.get(name)
            if canonical is None:
                raise ValueError(f"Unsupported Phase-C learner: {name}")
            if canonical not in self.active_model_names_:
                self.active_model_names_.append(canonical)
        if not self.active_model_names_:
            raise ValueError("At least one Phase-C learner must be enabled")
        scenario_cfg = phase_c.get("scenario_auxiliary", {})
        scenario_enabled = bool(scenario_cfg.get("enabled", True))
        phase_p_cfg = self.config.get("phase_p", {})
        phase_p_enabled = bool(phase_p_cfg.get("enabled", True))
        pilot_logistic = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=self.seed,
        ).fit(x_train, y_train)
        sample_weights = np.ones(len(x_train), dtype=float)
        weight_diagnostics: dict[str, Any] = {
            "mean": 1.0,
            "minimum": 1.0,
            "maximum": 1.0,
            "early_censored_fraction": 0.0,
            "validation_residual": 0.0,
            "class_residuals": [],
        }
        ipcw_enabled = bool(phase_p_cfg.get("ipcw_sample_reweighting_enabled", True))
        have_survival_arrays = all(
            value is not None
            for value in (train_times, train_events, val_times, val_events, cutpoints)
        )
        if phase_p_enabled and ipcw_enabled and have_survival_arrays:
            pilot_val = _complete_proba(pilot_logistic, x_val)
            sample_weights, weight_diagnostics = phase_p_ipcw_residual_weights(
                y_train,
                np.asarray(train_times, dtype=float),
                np.asarray(train_events, dtype=int),
                y_val,
                pilot_val,
                np.asarray(val_events, dtype=int),
                list(cutpoints or []),
                classification_calibration_mix=float(
                    phase_p_cfg.get("classification_calibration_mix", 0.5)
                ),
                residual_strength=float(phase_p_cfg.get("residual_weight_strength", 1.0)),
                minimum_positive_weight=float(phase_p_cfg.get("minimum_positive_weight", 0.25)),
                maximum_weight=float(phase_p_cfg.get("maximum_weight", 4.0)),
            )
            fitted_logistic = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                random_state=self.seed,
            ).fit(x_train, y_train, sample_weight=sample_weights)
        else:
            fitted_logistic = pilot_logistic
        if "logistic" in self.active_model_names_:
            self.models["logistic"] = fitted_logistic
        self.phase_p_sample_weights_ = sample_weights.copy()
        if "random_forest" in self.active_model_names_:
            rf_cfg = phase_c.get("random_forest", {})
            rf_base = RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 500)),
                max_depth=rf_cfg.get("max_depth", 10),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=self.seed,
                n_jobs=-1,
            ).fit(x_train, y_train, sample_weight=sample_weights)
            self.models["random_forest"] = CalibratedModel(rf_base).fit_calibrator(x_val, y_val)
        if "gradient_boosting_or_xgboost" in self.active_model_names_:
            self.models["gradient_boosting_or_xgboost"] = self._fit_boosting(
                x_train, y_train, sample_weights
            )
        mlp_cfg = phase_c.get("mlp", {})
        phase_e_loss_cfg = self.config.get("phase_e", {}).get("loss", {})
        configured_lambda_exp = float(phase_e_loss_cfg.get("lambda_exp", 0.3))
        phase_e_loss_enabled = bool(
            phase_e_loss_cfg.get("enabled", True)
            and (configured_lambda_exp <= 0.0 or cox_direction is not None)
        )
        model_selection_enabled = bool(
            phase_p_enabled and phase_p_cfg.get("model_selection_enabled", True)
        )
        if "mlp" in self.active_model_names_:
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
                sample_weights=sample_weights,
                scenario_train=scenario_train,
                scenario_val=scenario_val,
                scenario_loss_weight=float(scenario_cfg.get("loss_weight", 0.2))
                if scenario_enabled
                else 0.0,
                n_scenario_classes=int(scenario_cfg.get("n_actions", 6)) if scenario_enabled else 0,
                phase_p_model_selection_enabled=model_selection_enabled,
                phase_p_residual_weight=float(phase_p_cfg.get("checkpoint_residual_weight", 0.2))
                if model_selection_enabled
                else 0.0,
                val_events=np.asarray(val_events, dtype=int) if val_events is not None else None,
                phase_p_calibration_mix=float(phase_p_cfg.get("classification_calibration_mix", 0.5)),
                phase_e_lambda_cal=float(phase_e_loss_cfg.get("lambda_cal", 0.4))
                if phase_e_loss_enabled
                else 0.0,
                phase_e_lambda_exp=float(phase_e_loss_cfg.get("lambda_exp", 0.3))
                if phase_e_loss_enabled
                else 0.0,
                phase_e_lambda_clin=float(phase_e_loss_cfg.get("lambda_clin", 0.2))
                if phase_e_loss_enabled
                else 0.0,
                phase_e_kappa=float(phase_e_loss_cfg.get("tanh_kappa", 5.0)),
                cox_direction=cox_direction,
                clinical_monotonic_indices=list(
                    phase_e_loss_cfg.get(
                        "clinical_risk_state_indices", [12, 13, 14, 15, 16, 17, 25]
                    )
                ),
                clinical_monotonic_signs=list(
                    phase_e_loss_cfg.get("clinical_risk_state_signs", [1, 1, 1, 1, 1, 1, 1])
                ),
                seed=self.seed,
            )
        proba_stack = self.predict_base_stack(x_val)
        self.fusion_weights = optimize_brier_weights(proba_stack, y_val)
        stacked = self._fuse_stack(proba_stack)
        alpha = float(phase_c.get("fusion_alpha", 0.60))
        self.calibration_head = FusionCalibrationHead(alpha=alpha).fit(stacked, cluster_val, y_val)
        platt_enabled = bool(phase_p_enabled and phase_p_cfg.get("platt_calibration_enabled", True))
        if platt_enabled:
            before_platt = self._predict_before_phase_p(x_val, cluster_val)
            self.phase_p_platt = PhasePPlattCalibrator(random_state=self.seed).fit(before_platt, y_val)
        else:
            self.phase_p_platt = None
        self.mechanism_trace_ = {
            "active_learners": list(self.active_model_names_),
            "scenario_auxiliary_enabled": bool(
                isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                and self.models["mlp"].training_metadata.get("scenario_auxiliary_enabled", False)
            ),
            "scenario_loss_weight": float(
                self.models["mlp"].training_metadata.get("scenario_loss_weight", 0.0)
                if isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                else 0.0
            ),
            "phase_p_enabled": phase_p_enabled,
            "phase_p_ipcw_sample_reweighting": bool(
                phase_p_enabled and ipcw_enabled and have_survival_arrays
            ),
            "phase_p_model_selection": model_selection_enabled,
            "phase_p_platt_calibration": self.phase_p_platt is not None,
            "phase_p_weight_diagnostics": weight_diagnostics,
            "phase_e_differentiable_training": bool(
                isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                and self.models["mlp"].training_metadata.get(
                    "phase_e_differentiable_training", False
                )
            ),
            "phase_e_lambda_cal": float(
                self.models["mlp"].training_metadata.get("phase_e_lambda_cal", 0.0)
                if isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                else 0.0
            ),
            "phase_e_lambda_exp": float(
                self.models["mlp"].training_metadata.get("phase_e_lambda_exp", 0.0)
                if isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                else 0.0
            ),
            "phase_e_lambda_clin": float(
                self.models["mlp"].training_metadata.get("phase_e_lambda_clin", 0.0)
                if isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                else 0.0
            ),
            "phase_e_cox_direction_connected": bool(
                isinstance(self.models.get("mlp"), MLPClassifierWrapper)
                and self.models["mlp"].training_metadata.get("cox_direction_connected", False)
            ),
        }
        return self

    def _fit_boosting(
        self, x_train: np.ndarray, y_train: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> Any:
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
            ).fit(x_train, y_train, sample_weight=sample_weights)
        gb_cfg = phase_c.get("gradient_boosting_fallback", {})
        return GradientBoostingClassifier(
            n_estimators=int(gb_cfg.get("n_estimators", 500)),
            learning_rate=float(gb_cfg.get("learning_rate", 0.05)),
            max_depth=int(gb_cfg.get("max_depth", 3)),
            random_state=self.seed,
        ).fit(x_train, y_train, sample_weight=sample_weights)

    def predict_base_stack(self, x: np.ndarray) -> np.ndarray:
        probs = []
        for name in self.active_model_names_:
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

    def _predict_before_phase_p(self, x: np.ndarray, cluster_one_hot: np.ndarray) -> np.ndarray:
        stacked = self._fuse_stack(self.predict_base_stack(x))
        if self.calibration_head is not None:
            stacked = self.calibration_head.predict(stacked, cluster_one_hot)
        return stacked / np.clip(stacked.sum(axis=1, keepdims=True), 1e-12, None)

    def predict_proba(self, x: np.ndarray, cluster_one_hot: np.ndarray) -> np.ndarray:
        stacked = self._predict_before_phase_p(x, cluster_one_hot)
        if self.phase_p_platt is not None:
            stacked = self.phase_p_platt.predict(stacked)
        return stable_softmax(np.log(np.clip(stacked, 1e-12, 1.0)), axis=1)

    def predict_scenario_proba(self, x: np.ndarray) -> np.ndarray:
        model = self.models.get("mlp")
        if not isinstance(model, MLPClassifierWrapper):
            raise RuntimeError("MLP scenario auxiliary head is unavailable")
        return model.predict_scenario_proba(x)
