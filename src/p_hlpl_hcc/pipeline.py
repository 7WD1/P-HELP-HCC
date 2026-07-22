"""End-to-end P-HLPL-HCC training and inference pipeline."""

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
from .ensemble import PHlplEnsemble
from .metrics import classification_metrics
from .parallel import ParallelController, phase_p_residual_score
from .preprocessing import PHlplPreprocessor
from .society import SocietyTransformer


@dataclass
class PHlplHCCPipeline:
    config: dict[str, Any]
    seed: int = 42
    preprocessor: PHlplPreprocessor | None = None
    society: SocietyTransformer | None = None
    clusterer: PhenotypeClusterer | None = None
    ensemble: PHlplEnsemble | None = None
    cox: CoxElasticNetTorch | None = None
    cox_state_anchor: CoxElasticNetTorch | None = None
    counterfactual: CounterfactualSweep | None = None
    parallel_controller: ParallelController | None = None
    train_metrics_: dict[str, float] = field(default_factory=dict)
    val_metrics_: dict[str, float] = field(default_factory=dict)
    phase_e_loss_: dict[str, float] = field(default_factory=dict)
    mechanism_trace_: dict[str, Any] = field(default_factory=dict)
    phase_p_validation_residual_: float = 0.0
    selection_score_: float = 0.0
    cluster_feature_count_: int = 1
    state_agent_mask_: np.ndarray | None = None

    @staticmethod
    def agent_state_mask(phase_a_config: dict[str, Any]) -> np.ndarray:
        """Build the executable 46-state mask for named agent-drop variants."""

        mask = np.ones(46, dtype=np.float32)
        block_cfg = phase_a_config.get("agent_blocks", {})
        for block_name in ["patient", "tumor", "liver", "treatment", "guideline", "explanation"]:
            if not bool(block_cfg.get(block_name, True)):
                mask[SocietyTransformer.block_indices(block_name)] = 0.0
        return mask

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "PHlplHCCPipeline":
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
        self.preprocessor = PHlplPreprocessor(curated_dim=int(self.config["data"].get("curated_dim", 67)))
        x_train = self.preprocessor.fit_transform(train_df, y_train)
        x_val = self.preprocessor.transform(val_df)
        phase_a = self.config.get("phase_a", {})
        state_enabled = bool(phase_a.get("state_representation_enabled", True))
        if state_enabled:
            self.society = SocietyTransformer()
            s_train = self.society.fit_transform(x_train)
            s_val = self.society.transform(x_val)
            self.state_agent_mask_ = self.agent_state_mask(phase_a)
            s_train = s_train * self.state_agent_mask_
            s_val = s_val * self.state_agent_mask_
        else:
            self.society = None
            s_train, s_val = x_train, x_val
        cluster_cfg = self.config.get("phase_c", {}).get("clustering", {})
        if bool(cluster_cfg.get("enabled", True)):
            self.clusterer = PhenotypeClusterer(
                k=int(cluster_cfg.get("k", 4)),
                pca_variance=float(cluster_cfg.get("pca_variance", 0.90)),
                n_init=int(cluster_cfg.get("n_init", 20)),
                random_state=self.seed,
            ).fit(x_train)
            cluster_val = self.clusterer.one_hot(x_val)
            self.cluster_feature_count_ = int(cluster_val.shape[1])
        else:
            self.clusterer = None
            self.cluster_feature_count_ = 1
            cluster_val = np.ones((len(x_val), 1), dtype=float)
        phase_c_cfg = self.config.get("phase_c", {})
        scenario_aux_enabled = bool(
            phase_c_cfg.get("scenario_auxiliary", {}).get("enabled", True)
        )
        cf_cfg = phase_c_cfg.get("counterfactual", {})
        counterfactual_enabled = bool(
            cf_cfg.get("enabled", True) and self.society is not None
        )
        action_col = str(
            self.config.get("data", {}).get(
                "index_treatment_action_col", "index_treatment_action"
            )
        )
        train_actions = (
            extract_observed_actions_from_df(train_df, action_col=action_col)
            if scenario_aux_enabled or counterfactual_enabled
            else None
        )
        val_actions = (
            extract_observed_actions_from_df(val_df, action_col=action_col)
            if scenario_aux_enabled or counterfactual_enabled
            else None
        )
        train_times = train_df[self.config["data"]["target_time_col"]].to_numpy(dtype=float)
        train_events = train_df[self.config["data"]["event_col"]].to_numpy(dtype=int)
        val_times = val_df[self.config["data"]["target_time_col"]].to_numpy(dtype=float)
        val_events = val_df[self.config["data"]["event_col"]].to_numpy(dtype=int)
        survival_bins = self.config.get("data", {}).get(
            "survival_bins_months", [0, 6, 12, 24, 36, 48, 60, 72]
        )
        cutpoints = list(survival_bins[1:])
        cox_cfg = self.config.get("phase_e", {}).get("cox", {})
        phase_e_loss_cfg = self.config.get("phase_e", {}).get("loss", {})
        if (
            bool(cox_cfg.get("enabled", True))
            and bool(phase_e_loss_cfg.get("enabled", True))
            and float(phase_e_loss_cfg.get("lambda_exp", 0.3)) > 0.0
        ):
            self.cox_state_anchor = CoxElasticNetTorch(
                epochs=int(cox_cfg.get("epochs", 300)),
                learning_rate=float(cox_cfg.get("learning_rate", 0.03)),
                l1=float(cox_cfg.get("l1", 1e-3)),
                l2=float(cox_cfg.get("l2", 1e-3)),
            ).fit(s_train, train_times, train_events)
            cox_direction = self.cox_state_anchor.beta_
        else:
            self.cox_state_anchor = None
            cox_direction = None
        self.ensemble = PHlplEnsemble(self.config, seed=self.seed).fit(
            s_train,
            y_train,
            s_val,
            y_val,
            cluster_val,
            scenario_train=train_actions,
            scenario_val=val_actions,
            train_times=train_times,
            train_events=train_events,
            val_times=val_times,
            val_events=val_events,
            cutpoints=cutpoints,
            cox_direction=cox_direction,
        )
        cox_enabled = bool(cox_cfg.get("enabled", True))
        if cox_enabled:
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
        else:
            self.cox = None
        if counterfactual_enabled:
            if train_actions is None:
                raise RuntimeError("Counterfactual fitting requires recorded actions")
            self.counterfactual = CounterfactualSweep(
                actions=list(cf_cfg.get("actions", [])) or None,
                propensity_gate=tuple(cf_cfg.get("propensity_gate", [0.05, 0.95])),
                guideline_confidence_threshold=float(cf_cfg.get("guideline_confidence_threshold", 0.30)),
                bootstrap_replicates=int(cf_cfg.get("bootstrap_replicates", 200)),
                random_state=self.seed,
            ).fit(s_train, train_actions)
        else:
            self.counterfactual = None
        p_cfg = self.config.get("phase_p", {})
        phase_p_enabled = bool(p_cfg.get("enabled", True))
        if phase_p_enabled:
            self.parallel_controller = ParallelController(
                soft_error_threshold=float(p_cfg.get("soft_error_threshold", 0.18)),
                hard_error_threshold=float(p_cfg.get("hard_error_threshold", 0.32)),
                abstention_entropy_soft=float(p_cfg.get("abstention_entropy_soft", 0.65)),
                abstention_entropy_hard=float(p_cfg.get("abstention_entropy_hard", 0.85)),
                online_learning_rate=float(p_cfg.get("online_learning_rate", 5e-3)),
                proximal_weight=float(p_cfg.get("proximal_weight", 1e-2)),
                monitor_window=int(p_cfg.get("monitor_window", 30)),
                retrain_buffer=int(p_cfg.get("retrain_buffer", 200)),
                classification_calibration_mix=float(
                    p_cfg.get("classification_calibration_mix", 0.5)
                ),
            )
        else:
            self.parallel_controller = None
        mlp_metadata = getattr(self.ensemble.models.get("mlp"), "training_metadata", {})
        if bool(mlp_metadata.get("phase_e_differentiable_training", False)):
            pred_focal = float(mlp_metadata["phase_e_last_loss_pred_focal"])
            cal = float(mlp_metadata["phase_e_last_loss_cal"])
            exp = float(mlp_metadata["phase_e_last_loss_exp"])
            clin = float(mlp_metadata["phase_e_last_loss_clin"])
            lambda_cal = float(mlp_metadata["phase_e_lambda_cal"])
            lambda_exp = float(mlp_metadata["phase_e_lambda_exp"])
            lambda_clin = float(mlp_metadata["phase_e_lambda_clin"])
            self.phase_e_loss_ = {
                "last_loss_pred_class_weighted_focal": pred_focal,
                "last_loss_cal_scalar_weighted_brier": cal,
                "last_loss_exp_state_gradient_cox_direction": exp,
                "last_loss_clin_selected_gradient_sign": clin,
                "last_loss_total_without_scenario": (
                    pred_focal
                    + lambda_cal * cal
                    + lambda_exp * exp
                    + lambda_clin * clin
                ),
                "lambda_cal": lambda_cal,
                "lambda_exp": lambda_exp,
                "lambda_clin": lambda_clin,
            }
        else:
            self.phase_e_loss_ = {}
        self.train_metrics_ = self.evaluate(train_df)
        self.val_metrics_ = self.evaluate(val_df)
        model_selection_enabled = bool(
            phase_p_enabled and p_cfg.get("model_selection_enabled", True)
        )
        if model_selection_enabled:
            val_proba = self.predict_proba(val_df)
            self.phase_p_validation_residual_ = phase_p_residual_score(
                val_proba,
                y_val,
                val_events,
                classification_calibration_mix=float(
                    p_cfg.get("classification_calibration_mix", 0.5)
                ),
            )
        else:
            self.phase_p_validation_residual_ = 0.0
        candidate_residual_weight = (
            float(p_cfg.get("candidate_residual_weight", 0.1)) if model_selection_enabled else 0.0
        )
        self.selection_score_ = float(
            self.val_metrics_["macro_f1"]
            - candidate_residual_weight * self.phase_p_validation_residual_
        )
        self.mechanism_trace_ = dict(self.ensemble.mechanism_trace_)
        self.mechanism_trace_.update(
            {
                "counterfactual_reporting_enabled": self.counterfactual is not None,
                "scenario_auxiliary_role": "observed_action_shared_backbone_training_only",
                "scenario_survival_sweep_role": "forced_treatment_block_through_main_survival_predictor",
                "scenario_factual_arm_source": "recorded_pretreatment_action_required",
                "scenario_action_derived_auxiliary_handling": "neutralized_for_factual_and_forced_arms",
                "phase_p_controller_enabled": self.parallel_controller is not None,
                "phase_p_candidate_selection": model_selection_enabled,
                "phase_p_validation_residual": self.phase_p_validation_residual_,
                "candidate_selection_score": self.selection_score_,
                "state_representation_enabled": state_enabled,
                "phenotype_calibration_feature_enabled": self.clusterer is not None,
                "cox_explanation_anchor_enabled": self.cox is not None,
                "cox_state_direction_connected_to_training": self.cox_state_anchor is not None,
                "phase_e_training_scope": "mlp_branch_only",
                "phase_e_calibration_weight_scope": "one_scalar_phase_p_sample_weight_per_patient",
                "phase_e_explanation_proxy": "state_risk_input_gradient_vs_normalized_cox_direction",
                "phase_e_clinical_penalty": "selected_state_gradient_sign_hinge",
                "disabled_agent_blocks": [
                    name
                    for name in ["patient", "tumor", "liver", "treatment", "guideline", "explanation"]
                    if not bool(phase_a.get("agent_blocks", {}).get(name, True))
                ],
                "scenario_dynamic_rollout_enabled": False,
                "scenario_dynamic_rollout_reason": "physical_unit_dynamics_excluded_from_main_predict_path",
            }
        )
        return self

    def transform_state(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.preprocessor is None:
            raise RuntimeError("Pipeline is not fitted")
        time_col = self.config["data"]["target_time_col"]
        event_col = self.config["data"]["event_col"]
        label_col = self.config["data"]["label_col"]
        if time_col in df.columns and event_col in df.columns:
            prepared = validate_and_prepare_dataframe(df, time_col=time_col, event_col=event_col, label_col=label_col)
        else:
            prepared = df.copy()
        x = self.preprocessor.transform(prepared)
        state = self.society.transform(x) if self.society is not None else x
        if self.state_agent_mask_ is not None:
            state = state * self.state_agent_mask_
        return x, state

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.ensemble is None:
            raise RuntimeError("Pipeline is not fitted")
        x, state = self.transform_state(df)
        cluster = (
            self.clusterer.one_hot(x)
            if self.clusterer is not None
            else np.ones((len(x), self.cluster_feature_count_), dtype=float)
        )
        return self.ensemble.predict_proba(state, cluster)

    def predict_proba_from_state(
        self, state: np.ndarray, cluster_one_hot: np.ndarray | None = None
    ) -> np.ndarray:
        if self.ensemble is None:
            raise RuntimeError("Pipeline is not fitted")
        if cluster_one_hot is None:
            n_c = self.clusterer.kmeans.n_clusters if self.clusterer is not None else self.cluster_feature_count_
            cluster_one_hot = np.full((state.shape[0], n_c), 1.0 / n_c)
        return self.ensemble.predict_proba(state, cluster_one_hot)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(df), axis=1)

    def scenario_auxiliary_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return the shared-backbone auxiliary scenario probabilities."""

        if self.ensemble is None:
            raise RuntimeError("Pipeline is not fitted")
        _x, state = self.transform_state(df)
        return self.ensemble.predict_scenario_proba(state)

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

    def counterfactual_report(
        self,
        df: pd.DataFrame,
        row: int = 0,
        *,
        patient_bootstrap_predictions: dict[int, np.ndarray] | None = None,
        guideline_confidence_by_action: dict[int, float] | None = None,
    ) -> list[dict[str, object]]:
        if self.counterfactual is None:
            raise RuntimeError("Pipeline is not fitted")
        action_col = str(
            self.config.get("data", {}).get(
                "index_treatment_action_col", "index_treatment_action"
            )
        )
        observed_action = int(
            extract_observed_actions_from_df(
                df.iloc[[row]], action_col=action_col
            )[0]
        )
        x, state = self.transform_state(df)
        cluster_one_hot = (
            self.clusterer.one_hot(x[row : row + 1])
            if self.clusterer is not None
            else np.ones((1, self.cluster_feature_count_), dtype=float)
        )
        return self.counterfactual.sweep_patient(
            self,
            state[row],
            observed_action=observed_action,
            cluster_one_hot=cluster_one_hot,
            patient_bootstrap_predictions=patient_bootstrap_predictions,
            guideline_confidence_by_action=guideline_confidence_by_action,
        )

    def phenotype_quality(self, df: pd.DataFrame) -> dict[str, float]:
        if self.preprocessor is None:
            raise RuntimeError("Pipeline is not fitted")
        if self.clusterer is None:
            return {"silhouette": float("nan")}
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
        y = prepared[self.config["data"]["label_col"]].to_numpy(dtype=int)
        event = prepared[self.config["data"]["event_col"]].to_numpy(dtype=int)
        return [
            self.parallel_controller.observe(p, int(t), int(ev))
            for p, t, ev in zip(proba, y, event)
        ]
