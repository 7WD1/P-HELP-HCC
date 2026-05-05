"""End-to-end P-HELP-HCC training and inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .clustering import PhenotypeClusterer
from .constants import CANONICAL_COLUMNS, N_CLASSES
from .counterfactual import CounterfactualSweep, extract_observed_actions_from_df
from .cox import CoxElasticNetTorch
from .data import validate_and_prepare_dataframe
from .ensemble import PHelpEnsemble
from .explain import phase_e_loss_report
from .metrics import classification_metrics
from .parallel import ParallelController
from .preprocessing import PHelpPreprocessor
from .society import ArtificialSocietyDynamics, SocietyTransformer


@dataclass
class PHelpHCCPipeline:
    config: dict[str, Any]
    seed: int = 42
    preprocessor: PHelpPreprocessor | None = None
    society: SocietyTransformer | None = None
    clusterer: PhenotypeClusterer | None = None
    ensemble: PHelpEnsemble | None = None
    cox: CoxElasticNetTorch | None = None
    counterfactual: CounterfactualSweep | None = None
    parallel_controller: ParallelController | None = None
    train_metrics_: dict[str, float] = field(default_factory=dict)
    val_metrics_: dict[str, float] = field(default_factory=dict)
    phase_e_loss_: dict[str, float] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "PHelpHCCPipeline":
        train_df = validate_and_prepare_dataframe(
            train_df,
            time_col=self.config["data"]["target_time_col"],
            event_col=self.config["data"]["event_col"],
            label_col=self.config["data"]["label_col"],
        )
        val_df = validate_and_prepare_dataframe(
            val_df,
            time_col=self.config["data"]["target_time_col"],
            event_col=self.config["data"]["event_col"],
            label_col=self.config["data"]["label_col"],
        )
        y_train = train_df[self.config["data"]["label_col"]].to_numpy(dtype=int)
        y_val = val_df[self.config["data"]["label_col"]].to_numpy(dtype=int)
        self.preprocessor = PHelpPreprocessor(curated_dim=int(self.config["data"].get("curated_dim", 67)))
        x_train = self.preprocessor.fit_transform(train_df, y_train)
        x_val = self.preprocessor.transform(val_df)
        phase_a = self.config.get("phase_a", {})
        self.society = SocietyTransformer(process_noise_std=float(phase_a.get("process_noise_std", 0.05)))
        s_train = self.society.fit_transform(x_train)
        s_val = self.society.transform(x_val)
        cluster_cfg = self.config.get("phase_c", {}).get("clustering", {})
        self.clusterer = PhenotypeClusterer(
            k=int(cluster_cfg.get("k", 4)),
            pca_variance=float(cluster_cfg.get("pca_variance", 0.90)),
            n_init=int(cluster_cfg.get("n_init", 20)),
            random_state=self.seed,
        ).fit(x_train)
        cluster_val = self.clusterer.one_hot(x_val)
        self.ensemble = PHelpEnsemble(self.config, seed=self.seed).fit(s_train, y_train, s_val, y_val, cluster_val)
        cox_cfg = self.config.get("phase_e", {}).get("cox", {})
        self.cox = CoxElasticNetTorch(
            epochs=int(cox_cfg.get("epochs", 300)),
            learning_rate=float(cox_cfg.get("learning_rate", 0.03)),
            l1=float(cox_cfg.get("l1", 1e-3)),
            l2=float(cox_cfg.get("l2", 1e-3)),
        ).fit(
            x_train,
            train_df[self.config["data"]["target_time_col"]].to_numpy(dtype=float),
            train_df[self.config["data"]["event_col"]].to_numpy(dtype=int),
        )
        cf_cfg = self.config.get("phase_c", {}).get("counterfactual", {})
        self.counterfactual = CounterfactualSweep(
            actions=list(cf_cfg.get("actions", [])) or None,
            propensity_gate=tuple(cf_cfg.get("propensity_gate", [0.05, 0.95])),
            guideline_confidence_threshold=float(cf_cfg.get("guideline_confidence_threshold", 0.30)),
            bootstrap_replicates=int(cf_cfg.get("bootstrap_replicates", 200)),
            random_state=self.seed,
            dynamics=ArtificialSocietyDynamics(
                delta_t_months=float(phase_a.get("delta_t_months", 1)),
                horizon_months=int(phase_a.get("horizon_months", 72)),
                d_max_cm=float(phase_a.get("d_max_cm", 20.0)),
                afp_alpha_per_month=float(phase_a.get("afp_alpha_per_month", 0.04)),
                afp_beta=float(phase_a.get("afp_beta", 1.8)),
                fibrosis_age_rate_per_year=float(phase_a.get("fibrosis_age_rate_per_year", 0.002)),
                fibrosis_treatment_bump=float(phase_a.get("fibrosis_treatment_bump", 0.015)),
                fibrosis_recovery_rate=float(phase_a.get("fibrosis_recovery_rate", 0.020)),
                process_noise_std=0.0,
            ),
        ).fit(s_train, extract_observed_actions_from_df(train_df))
        p_cfg = self.config.get("phase_p", {})
        self.parallel_controller = ParallelController(
            soft_error_threshold=float(p_cfg.get("soft_error_threshold", 0.18)),
            hard_error_threshold=float(p_cfg.get("hard_error_threshold", 0.32)),
            abstention_entropy_soft=float(p_cfg.get("abstention_entropy_soft", 0.65)),
            abstention_entropy_hard=float(p_cfg.get("abstention_entropy_hard", 0.85)),
            online_learning_rate=float(p_cfg.get("online_learning_rate", 5e-3)),
            proximal_weight=float(p_cfg.get("proximal_weight", 1e-2)),
            monitor_window=int(p_cfg.get("monitor_window", 30)),
            retrain_buffer=int(p_cfg.get("retrain_buffer", 200)),
            classification_calibration_mix=float(p_cfg.get("classification_calibration_mix", 0.5)),
        )
        train_proba = self.predict_proba(train_df)
        shap_proxy = x_train * self.cox.beta_.reshape(1, -1)
        loss_cfg = self.config.get("phase_e", {}).get("loss", {})
        survival_bins = self.config.get("data", {}).get("survival_bins_months", [0, 6, 12, 24, 36, 48, 60, 72])
        cutpoints = list(survival_bins[1:])
        self.phase_e_loss_ = phase_e_loss_report(
            train_proba,
            y_train,
            shap_proxy,
            self.cox.beta_,
            x_train,
            self.preprocessor.output_names_,
            times=train_df[self.config["data"]["target_time_col"]].to_numpy(dtype=float),
            events=train_df[self.config["data"]["event_col"]].to_numpy(dtype=int),
            cutpoints=cutpoints,
            lambda_cal=float(loss_cfg.get("lambda_cal", 0.4)),
            lambda_exp=float(loss_cfg.get("lambda_exp", 0.3)),
            lambda_clin=float(loss_cfg.get("lambda_clin", 0.2)),
            kappa=float(loss_cfg.get("tanh_kappa", 5.0)),
        )
        self.train_metrics_ = self.evaluate(train_df)
        self.val_metrics_ = self.evaluate(val_df)
        return self

    def transform_state(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.preprocessor is None or self.society is None:
            raise RuntimeError("Pipeline is not fitted")
        time_col = self.config["data"]["target_time_col"]
        event_col = self.config["data"]["event_col"]
        label_col = self.config["data"]["label_col"]
        if time_col in df.columns and event_col in df.columns:
            prepared = validate_and_prepare_dataframe(df, time_col=time_col, event_col=event_col, label_col=label_col)
        else:
            prepared = df.copy()
        x = self.preprocessor.transform(prepared)
        return x, self.society.transform(x)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.ensemble is None or self.clusterer is None:
            raise RuntimeError("Pipeline is not fitted")
        x, state = self.transform_state(df)
        cluster = self.clusterer.one_hot(x)
        return self.ensemble.predict_proba(state, cluster)

    def predict_proba_from_state(
        self, state: np.ndarray, cluster_one_hot: np.ndarray | None = None
    ) -> np.ndarray:
        if self.ensemble is None or self.clusterer is None:
            raise RuntimeError("Pipeline is not fitted")
        if cluster_one_hot is None:
            n_c = self.clusterer.kmeans.n_clusters
            cluster_one_hot = np.full((state.shape[0], n_c), 1.0 / n_c)
        return self.ensemble.predict_proba(state, cluster_one_hot)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(df), axis=1)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        prepared = validate_and_prepare_dataframe(
            df,
            time_col=self.config["data"]["target_time_col"],
            event_col=self.config["data"]["event_col"],
            label_col=self.config["data"]["label_col"],
        )
        y = prepared[self.config["data"]["label_col"]].to_numpy(dtype=int)
        proba = self.predict_proba(prepared)
        return classification_metrics(
            y,
            proba,
            times=prepared[self.config["data"]["target_time_col"]].to_numpy(dtype=float),
            events=prepared[self.config["data"]["event_col"]].to_numpy(dtype=int),
        )

    def counterfactual_report(self, df: pd.DataFrame, row: int = 0) -> list[dict[str, float | str | bool]]:
        if self.counterfactual is None or self.clusterer is None:
            raise RuntimeError("Pipeline is not fitted")
        x, state = self.transform_state(df)
        cluster_one_hot = self.clusterer.one_hot(x[row : row + 1])
        return self.counterfactual.sweep_patient(self, state[row], cluster_one_hot=cluster_one_hot)

    def phenotype_quality(self, df: pd.DataFrame) -> dict[str, float]:
        if self.clusterer is None:
            raise RuntimeError("Pipeline is not fitted")
        x, _state = self.transform_state(df)
        return self.clusterer.quality(x)

    def phase_p_observe(self, df: pd.DataFrame) -> list[dict[str, float | str]]:
        if self.parallel_controller is None:
            raise RuntimeError("Pipeline is not fitted")
        prepared = validate_and_prepare_dataframe(
            df,
            time_col=self.config["data"]["target_time_col"],
            event_col=self.config["data"]["event_col"],
            label_col=self.config["data"]["label_col"],
        )
        proba = self.predict_proba(prepared)
        pred = np.argmax(proba, axis=1)
        y = prepared[self.config["data"]["label_col"]].to_numpy(dtype=int)
        event = prepared[self.config["data"]["event_col"]].to_numpy(dtype=int)
        cumulative_incidence = proba[:, :2].sum(axis=1)
        return [
            self.parallel_controller.observe(int(p), int(t), float(ci), int(ev))
            for p, t, ci, ev in zip(pred, y, cumulative_incidence, event)
        ]
