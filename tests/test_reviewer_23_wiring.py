import sys
import unittest
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.baseline import CoxPHSurvivalPipeline
from p_hlpl_hcc.config import apply_fast_overrides, apply_named_ablation, apply_named_variant, load_config
from p_hlpl_hcc.data import generate_fixture_hcc_records, validate_and_prepare_dataframe
from p_hlpl_hcc.ensemble import PHlplEnsemble, PhasePPlattCalibrator
from p_hlpl_hcc.neural import train_mlp_classifier
from p_hlpl_hcc.parallel import ParallelController, phase_p_ipcw_residual_weights
from p_hlpl_hcc.parallel import phase_p_residuals
from p_hlpl_hcc.pipeline import PHlplHCCPipeline
from p_hlpl_hcc.society import SocietyTransformer


class Reviewer23WiringTests(unittest.TestCase):
    def test_one_command_workflow_includes_each_named_phase_p_ablation(self):
        workflow = runpy.run_path(str(ROOT / "scripts" / "reproduce.py"))
        named_ablations = workflow["NAMED_ABLATIONS"]
        self.assertTrue(
            {
                "PhasePNoIPCW",
                "PhasePNoCheckpoint",
                "PhasePNoPlatt",
            }.issubset(named_ablations)
        )

    def test_named_a5_a6_configs_disable_the_claimed_components(self):
        base = load_config(ROOT / "configs" / "default.yaml")
        self.assertFalse(base["phase_a"]["tumor_update_enabled"])
        self.assertFalse(base["phase_a"]["fibrosis_update_enabled"])
        self.assertNotIn("process_noise_std", base["phase_a"])
        dp = base["safeguards"]["federated"]["dp_sgd"]
        self.assertEqual(dp["epsilon_per_round"], 0.4)
        self.assertEqual(dp["epsilon_total_design_budget"], 4.0)
        self.assertEqual(dp["planned_rounds"], 100)
        manifest = ROOT / "configs" / "ablations.yaml"
        full = apply_named_ablation(base, manifest, "full")
        a5 = apply_named_ablation(base, manifest, "A5")
        a6 = apply_named_ablation(base, manifest, "a6")

        self.assertTrue(full["phase_c"]["scenario_auxiliary"]["enabled"])
        self.assertTrue(full["phase_p"]["enabled"])
        self.assertFalse(a5["phase_c"]["scenario_auxiliary"]["enabled"])
        self.assertTrue(a5["phase_c"]["counterfactual"]["enabled"])
        self.assertTrue(a5["phase_p"]["enabled"])
        self.assertTrue(a6["phase_c"]["scenario_auxiliary"]["enabled"])
        self.assertFalse(a6["phase_p"]["enabled"])
        self.assertFalse(a6["phase_p"]["ipcw_sample_reweighting_enabled"])
        self.assertFalse(a6["phase_p"]["model_selection_enabled"])
        self.assertFalse(a6["phase_p"]["platt_calibration_enabled"])
        a4 = apply_named_ablation(base, manifest, "A4")
        self.assertFalse(a4["phase_e"]["loss"]["enabled"])

    def test_all_named_ablation_and_dynamics_configs_resolve(self):
        base = load_config(ROOT / "configs" / "default.yaml")
        manifest = ROOT / "configs" / "ablations.yaml"
        names = [
            "A1", "A2", "A3", "A4", "A5", "A6",
            "PhasePNoIPCW", "PhasePNoCheckpoint", "PhasePNoPlatt",
        ]
        resolved = {name: apply_named_ablation(base, manifest, name) for name in names}
        self.assertEqual(resolved["A1"]["experiment"]["pipeline"], "coxph_baseline")
        self.assertEqual(resolved["A1"]["phase_e"]["cox"]["l1"], 0.0)
        self.assertEqual(resolved["A1"]["phase_e"]["cox"]["l2"], 0.0)
        self.assertEqual(len(resolved["A2"]["phase_c"]["learners"]), 3)
        self.assertFalse(resolved["A3"]["phase_c"]["clustering"]["enabled"])
        self.assertTrue(resolved["A4"]["phase_c"]["clustering"]["enabled"])
        dynamics = ROOT / "configs" / "dynamics.yaml"
        variants = {
            name: apply_named_variant(base, dynamics, name, experiment_key="dynamics_variant")
            for name in ["static_only", "gompertz_only", "fibrosis_only", "full_dynamics"]
        }
        self.assertFalse(variants["static_only"]["phase_a"]["tumor_update_enabled"])
        self.assertTrue(variants["full_dynamics"]["phase_a"]["fibrosis_update_enabled"])
        phase_p_expected = {
            "PhasePNoIPCW": (False, True, True),
            "PhasePNoCheckpoint": (True, False, True),
            "PhasePNoPlatt": (True, True, False),
        }
        for name, expected in phase_p_expected.items():
            phase_p = resolved[name]["phase_p"]
            self.assertTrue(phase_p["enabled"])
            self.assertEqual(
                (
                    phase_p["ipcw_sample_reweighting_enabled"],
                    phase_p["model_selection_enabled"],
                    phase_p["platt_calibration_enabled"],
                ),
                expected,
            )
        for name, block in [
            ("DropPatient", "patient"),
            ("DropTumor", "tumor"),
            ("DropLiver", "liver"),
            ("DropTreatment", "treatment"),
            ("DropGuideline", "guideline"),
            ("DropExplanation", "explanation"),
        ]:
            self.assertFalse(apply_named_ablation(base, manifest, name)["phase_a"]["agent_blocks"][block])
        self.assertEqual(apply_named_ablation(base, manifest, "NoCal")["phase_e"]["loss"]["lambda_cal"], 0.0)
        self.assertEqual(apply_named_ablation(base, manifest, "NoExp")["phase_e"]["loss"]["lambda_exp"], 0.0)
        self.assertEqual(apply_named_ablation(base, manifest, "NoClin")["phase_e"]["loss"]["lambda_clin"], 0.0)
        pred_only = apply_named_ablation(base, manifest, "PredOnlyFocal")
        self.assertEqual(
            [
                pred_only["phase_e"]["loss"]["lambda_cal"],
                pred_only["phase_e"]["loss"]["lambda_exp"],
                pred_only["phase_e"]["loss"]["lambda_clin"],
            ],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            pred_only["experiment"]["loss_ablation_label"],
            "prediction_only_class_weighted_focal",
        )
        patient_only = apply_named_ablation(base, manifest, "PatientOnly")
        self.assertEqual(
            patient_only["phase_a"]["agent_blocks"],
            {
                "patient": True,
                "tumor": False,
                "liver": False,
                "treatment": False,
                "guideline": False,
                "explanation": False,
            },
        )
        patient_mask = PHlplHCCPipeline.agent_state_mask(patient_only["phase_a"])
        patient_slice = SocietyTransformer.block_indices("patient")
        self.assertTrue(np.all(patient_mask[patient_slice] == 1.0))
        self.assertEqual(int(np.sum(patient_mask)), patient_slice.stop - patient_slice.start)
        drop_tumor = apply_named_ablation(base, manifest, "DropTumor")
        mask = PHlplHCCPipeline.agent_state_mask(drop_tumor["phase_a"])
        tumor_slice = SocietyTransformer.block_indices("tumor")
        self.assertTrue(np.all(mask[tumor_slice] == 0.0))
        self.assertEqual(int(np.sum(mask == 0.0)), tumor_slice.stop - tumor_slice.start)

    def test_a1_coxph_pipeline_produces_eight_ordered_probabilities(self):
        config = apply_fast_overrides(load_config(ROOT / "configs" / "default.yaml"))
        config = apply_named_ablation(config, ROOT / "configs" / "ablations.yaml", "A1")
        df = validate_and_prepare_dataframe(generate_fixture_hcc_records(n=48, seed=22))
        model = CoxPHSurvivalPipeline(config=config, seed=3).fit(df.iloc[:36], df.iloc[36:42])
        proba = model.predict_proba(df.iloc[42:])
        self.assertEqual(proba.shape, (6, 8))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_a5_switch_changes_shared_mlp_training_and_output(self):
        rng = np.random.default_rng(9)
        x_train = rng.normal(size=(48, 6))
        x_val = rng.normal(size=(24, 6))
        y_train = np.arange(48) % 8
        y_val = np.arange(24) % 8
        scenario_train = (x_train[:, 0] > 0).astype(int) + (x_train[:, 1] > 0).astype(int)
        scenario_val = (x_val[:, 0] > 0).astype(int) + (x_val[:, 1] > 0).astype(int)
        common = dict(
            hidden_dims=[12],
            dropout=0.0,
            learning_rate=0.01,
            weight_decay=0.0,
            batch_size=12,
            epochs=3,
            patience=3,
            gamma=1.5,
            class_weights=[1.0] * 8,
            seed=17,
        )
        full = train_mlp_classifier(
            x_train,
            y_train,
            x_val,
            y_val,
            scenario_train=scenario_train,
            scenario_val=scenario_val,
            scenario_loss_weight=1.0,
            n_scenario_classes=3,
            **common,
        )
        a5 = train_mlp_classifier(
            x_train,
            y_train,
            x_val,
            y_val,
            scenario_loss_weight=0.0,
            n_scenario_classes=0,
            **common,
        )
        self.assertTrue(full.training_metadata["scenario_auxiliary_enabled"])
        self.assertFalse(a5.training_metadata["scenario_auxiliary_enabled"])
        self.assertEqual(full.predict_scenario_proba(x_val).shape, (len(x_val), 3))
        with self.assertRaises(RuntimeError):
            a5.predict_scenario_proba(x_val)
        self.assertFalse(np.allclose(full.predict_proba(x_val), a5.predict_proba(x_val)))

    def test_phase_e_losses_are_differentiable_and_change_training(self):
        rng = np.random.default_rng(18)
        x_train = rng.normal(size=(40, 6))
        x_val = rng.normal(size=(16, 6))
        y_train = np.arange(40) % 8
        y_val = np.arange(16) % 8
        common = dict(
            hidden_dims=[10], dropout=0.0, learning_rate=0.01, weight_decay=0.0,
            batch_size=10, epochs=2, patience=2, gamma=1.5,
            class_weights=[1.0] * 8, seed=21,
        )
        plain = train_mlp_classifier(x_train, y_train, x_val, y_val, **common)
        phase_e = train_mlp_classifier(
            x_train, y_train, x_val, y_val,
            phase_e_lambda_cal=0.4,
            phase_e_lambda_exp=0.3,
            phase_e_lambda_clin=0.2,
            cox_direction=np.array([0.4, -0.2, 0.1, 0.3, -0.1, 0.2]),
            clinical_monotonic_indices=[0, 2, 3],
            clinical_monotonic_signs=[1, 1, 1],
            **common,
        )
        self.assertTrue(phase_e.training_metadata["phase_e_differentiable_training"])
        self.assertTrue(phase_e.training_metadata["cox_direction_connected"])
        self.assertGreaterEqual(phase_e.training_metadata["phase_e_last_loss_cal"], 0.0)
        self.assertFalse(np.allclose(plain.predict_proba(x_val), phase_e.predict_proba(x_val)))

    def test_phase_p_residual_matches_probability_equation(self):
        probabilities = np.array([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
        truth = np.array([1, 0])
        observed = phase_p_residuals(probabilities, truth, np.array([1, 0]), classification_calibration_mix=0.5)
        target = np.eye(3)[truth]
        expected = 1.0 - probabilities[np.arange(2), truth] + 0.5 * np.square(probabilities - target).sum(axis=1)
        self.assertTrue(np.allclose(observed, expected))

    def test_phase_p_soft_update_matches_manuscript_proximal_equation(self):
        controller = ParallelController(
            online_learning_rate=0.2,
            proximal_weight=0.1,
        )
        weights = np.array([0.55, 0.30, 0.15])
        gradient = np.array([0.20, -0.10, -0.10])
        anchor = np.array([0.40, 0.35, 0.25])
        expected = weights - 0.2 * gradient - 0.1 * (weights - anchor)
        expected = np.maximum(expected, 0.0)
        expected /= expected.sum()

        observed = controller.soft_update_fusion_weights(weights, gradient, anchor)

        self.assertTrue(np.allclose(observed, expected))

    def test_phase_p_ipcw_weights_are_residual_and_censoring_informed(self):
        train_classes = np.arange(16) % 8
        train_times = np.array([4, 8, 13, 20, 28, 40, 55, 80] * 2, dtype=float)
        train_events = np.array([1, 1, 1, 0, 1, 0, 1, 0] * 2, dtype=int)
        val_classes = np.arange(16) % 8
        val_events = np.array([1, 1, 1, 0, 1, 0, 1, 0] * 2, dtype=int)
        val_proba = np.full((16, 8), 0.05)
        val_proba[np.arange(16), val_classes] = np.linspace(0.25, 0.60, 16)
        val_proba /= val_proba.sum(axis=1, keepdims=True)
        weights, diagnostics = phase_p_ipcw_residual_weights(
            train_classes,
            train_times,
            train_events,
            val_classes,
            val_proba,
            val_events,
            [6, 12, 24, 36, 48, 60, 72],
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=8)
        self.assertGreater(diagnostics["early_censored_fraction"], 0.0)
        self.assertEqual(len(diagnostics["class_residuals"]), 8)
        self.assertFalse(np.allclose(weights, np.ones_like(weights)))

    def test_a6_switch_removes_platt_from_the_prediction_path(self):
        raw = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.60, 0.25, 0.15],
                [0.20, 0.65, 0.15],
                [0.15, 0.70, 0.15],
                [0.15, 0.20, 0.65],
                [0.10, 0.15, 0.75],
            ]
        )
        y = np.array([0, 1, 1, 1, 2, 2])
        calibrator = PhasePPlattCalibrator().fit(raw, y)
        full = PHlplEnsemble(config={})
        a6 = PHlplEnsemble(config={})
        full.phase_p_platt = calibrator
        full._predict_before_phase_p = lambda _x, _cluster: raw
        a6._predict_before_phase_p = lambda _x, _cluster: raw
        dummy = np.zeros((len(raw), 1))
        cluster = np.ones((len(raw), 1))
        full_proba = full.predict_proba(dummy, cluster)
        a6_proba = a6.predict_proba(dummy, cluster)
        self.assertTrue(np.allclose(full_proba.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(a6_proba, raw))
        self.assertFalse(np.allclose(full_proba, a6_proba))

    def test_independent_phase_p_ablation_configs_execute_the_claimed_paths(self):
        base = apply_fast_overrides(load_config(ROOT / "configs" / "default.yaml"))
        manifest = ROOT / "configs" / "ablations.yaml"
        rng = np.random.default_rng(43)
        x_train = rng.normal(size=(64, 10))
        x_val = rng.normal(size=(24, 10))
        y_train = np.arange(64) % 8
        y_val = np.arange(24) % 8
        scenario_train = np.arange(64) % 6
        scenario_val = np.arange(24) % 6
        cluster_val = np.eye(4)[np.arange(24) % 4]
        times_train = np.linspace(3, 90, 64)
        times_val = np.linspace(3, 90, 24)
        events_train = np.ones(64, dtype=int)
        events_val = np.ones(24, dtype=int)
        expected = {
            "PhasePNoIPCW": (False, True, True),
            "PhasePNoCheckpoint": (True, False, True),
            "PhasePNoPlatt": (True, True, False),
        }
        for offset, (name, flags) in enumerate(expected.items()):
            config = apply_named_ablation(base, manifest, name)
            config["phase_c"]["learners"] = ["logistic", "mlp"]
            config["phase_c"]["mlp"].update(
                {"hidden_dims": [10], "epochs": 1, "patience": 1, "batch_size": 16}
            )
            config["phase_e"]["loss"]["enabled"] = False
            model = PHlplEnsemble(config=config, seed=30 + offset).fit(
                x_train,
                y_train,
                x_val,
                y_val,
                cluster_val,
                scenario_train=scenario_train,
                scenario_val=scenario_val,
                train_times=times_train,
                train_events=events_train,
                val_times=times_val,
                val_events=events_val,
                cutpoints=[6, 12, 24, 36, 48, 60, 72],
            )
            trace = model.mechanism_trace_
            self.assertEqual(
                (
                    trace["phase_p_ipcw_sample_reweighting"],
                    trace["phase_p_model_selection"],
                    trace["phase_p_platt_calibration"],
                ),
                flags,
            )

    def test_a6_executable_config_disables_all_phase_p_fit_paths(self):
        base = load_config(ROOT / "configs" / "default.yaml")
        config = apply_named_ablation(base, ROOT / "configs" / "ablations.yaml", "A6")
        config["phase_c"]["xgboost"]["enabled_if_installed"] = False
        config["phase_c"]["random_forest"].update({"n_estimators": 5, "max_depth": 3})
        config["phase_c"]["gradient_boosting_fallback"].update(
            {"n_estimators": 5, "max_depth": 2}
        )
        config["phase_c"]["mlp"].update(
            {"hidden_dims": [10], "epochs": 2, "patience": 2, "batch_size": 16}
        )
        rng = np.random.default_rng(31)
        x_train = rng.normal(size=(64, 10))
        x_val = rng.normal(size=(24, 10))
        y_train = np.arange(64) % 8
        y_val = np.arange(24) % 8
        scenario_train = np.arange(64) % 6
        scenario_val = np.arange(24) % 6
        cluster_val = np.eye(4)[np.arange(24) % 4]
        times_train = np.linspace(3, 80, 64)
        times_val = np.linspace(3, 80, 24)
        events_train = (np.arange(64) % 4 != 0).astype(int)
        events_val = (np.arange(24) % 4 != 0).astype(int)
        model = PHlplEnsemble(config=config, seed=5).fit(
            x_train,
            y_train,
            x_val,
            y_val,
            cluster_val,
            scenario_train=scenario_train,
            scenario_val=scenario_val,
            train_times=times_train,
            train_events=events_train,
            val_times=times_val,
            val_events=events_val,
            cutpoints=[6, 12, 24, 36, 48, 60, 72],
        )
        self.assertFalse(model.mechanism_trace_["phase_p_enabled"])
        self.assertFalse(model.mechanism_trace_["phase_p_ipcw_sample_reweighting"])
        self.assertFalse(model.mechanism_trace_["phase_p_model_selection"])
        self.assertFalse(model.mechanism_trace_["phase_p_platt_calibration"])
        self.assertIsNone(model.phase_p_platt)
        self.assertTrue(np.allclose(model.phase_p_sample_weights_, 1.0))
        raw = model._predict_before_phase_p(x_val, cluster_val)
        self.assertTrue(np.allclose(model.predict_proba(x_val, cluster_val), raw))


if __name__ == "__main__":
    unittest.main()
