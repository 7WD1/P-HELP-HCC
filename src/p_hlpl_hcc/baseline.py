"""Executable Cox proportional-hazards baseline for named ablation A1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cox import CoxElasticNetTorch
from .data import validate_and_prepare_dataframe
from .metrics import classification_metrics
from .preprocessing import PHlplPreprocessor


@dataclass
class CoxPHSurvivalPipeline:
    """Cox PH model with Breslow baseline mapped to eight survival bins."""

    config: dict[str, Any]
    seed: int = 42
    preprocessor: PHlplPreprocessor | None = None
    cox: CoxElasticNetTorch | None = None
    cutpoints_: list[float] = field(default_factory=list)
    baseline_cumulative_hazard_: np.ndarray | None = None
    mechanism_trace_: dict[str, Any] = field(default_factory=dict)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config["data"]
        return validate_and_prepare_dataframe(
            df,
            time_col=cfg["target_time_col"],
            event_col=cfg["event_col"],
            label_col=cfg["label_col"],
            require_unambiguous_hard_labels=False,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "CoxPHSurvivalPipeline":
        train = self._prepare(train_df)
        cfg = self.config["data"]
        time_col, event_col = cfg["target_time_col"], cfg["event_col"]
        bins = list(map(float, cfg.get("survival_bins_months", [0, 6, 12, 24, 36, 48, 60, 72])))
        self.cutpoints_ = bins[1:] if bins and bins[0] == 0 else bins
        self.preprocessor = PHlplPreprocessor(curated_dim=int(cfg.get("curated_dim", 67)))
        x = self.preprocessor.fit_transform(train)
        cox_cfg = self.config.get("phase_e", {}).get("cox", {})
        self.cox = CoxElasticNetTorch(
            epochs=int(cox_cfg.get("epochs", 300)),
            learning_rate=float(cox_cfg.get("learning_rate", 0.03)),
            l1=float(cox_cfg.get("l1", 1e-3)),
            l2=float(cox_cfg.get("l2", 1e-3)),
        ).fit(x, train[time_col].to_numpy(float), train[event_col].to_numpy(int))
        log_risk = self.cox.predict_log_hazard(x)
        time = train[time_col].to_numpy(float)
        event = train[event_col].to_numpy(int)
        event_times = np.sort(np.unique(time[event == 1]))
        increments: list[tuple[float, float]] = []
        exp_risk = np.exp(np.clip(log_risk, -30, 30))
        for event_time in event_times:
            deaths = int(np.sum((time == event_time) & (event == 1)))
            denominator = float(exp_risk[time >= event_time].sum())
            if denominator > 0:
                increments.append((float(event_time), deaths / denominator))
        self.baseline_cumulative_hazard_ = np.array(
            [sum(value for event_time, value in increments if event_time <= cut) for cut in self.cutpoints_],
            dtype=float,
        )
        self.mechanism_trace_ = {
            "named_ablation": "A1",
            "pipeline": "coxph_breslow_only",
            "uses_deep_encoder": False,
            "uses_phenotype_calibration_feature": False,
            "uses_scenario_head": False,
            "uses_phase_p": False,
        }
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.cox is None or self.baseline_cumulative_hazard_ is None:
            raise RuntimeError("CoxPHSurvivalPipeline is not fitted")
        prepared = self._prepare(df)
        x = self.preprocessor.transform(prepared)
        relative_risk = np.exp(np.clip(self.cox.predict_log_hazard(x), -30, 30))
        survival = np.exp(-relative_risk[:, None] * self.baseline_cumulative_hazard_[None, :])
        probabilities = np.column_stack(
            [1.0 - survival[:, 0], *[survival[:, idx - 1] - survival[:, idx] for idx in range(1, len(self.cutpoints_))], survival[:, -1]]
        )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(df), axis=1)

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        prepared = self._prepare(df)
        cfg = self.config["data"]
        label_col, time_col, event_col = cfg["label_col"], cfg["target_time_col"], cfg["event_col"]
        proba = self.predict_proba(prepared)
        auditable = prepared[label_col].notna().to_numpy()
        result: dict[str, Any] = {
            "n_rows": int(len(prepared)),
            "n_hard_endpoint_rows": int(auditable.sum()),
        }
        if auditable.any():
            result.update(
                classification_metrics(
                    prepared.loc[auditable, label_col].to_numpy(dtype=int),
                    proba[auditable],
                    times=prepared.loc[auditable, time_col].to_numpy(float),
                    events=prepared.loc[auditable, event_col].to_numpy(int),
                )
            )
        return result

    def phenotype_quality(self, df: pd.DataFrame) -> dict[str, float]:
        return {"silhouette": float("nan")}
