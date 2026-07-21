import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.config import apply_fast_overrides, load_config
from p_hlpl_hcc.data import generate_fixture_hcc_records, validate_and_prepare_dataframe
from p_hlpl_hcc.pipeline import PHlplHCCPipeline
from p_hlpl_hcc.society import SocietyTransformer


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_fit_predict_counterfactual(self):
        config = apply_fast_overrides(load_config(ROOT / "configs" / "default.yaml"))
        df = validate_and_prepare_dataframe(generate_fixture_hcc_records(n=72, seed=11))
        train = df.iloc[:48].reset_index(drop=True)
        val = df.iloc[48:60].reset_index(drop=True)
        test = df.iloc[60:].reset_index(drop=True)
        model = PHlplHCCPipeline(config=config, seed=42).fit(train, val)
        proba = model.predict_proba(test)
        self.assertEqual(proba.shape, (len(test), 8))
        self.assertTrue(((proba.sum(axis=1) - 1.0) ** 2).max() < 1e-8)
        inference_only = test.drop(columns=["overall_survival_months", "event", "survival_class"])
        infer_proba = model.predict_proba(inference_only)
        self.assertEqual(infer_proba.shape, (len(test), 8))
        scenario_proba = model.scenario_auxiliary_proba(inference_only)
        self.assertEqual(scenario_proba.shape, (len(test), 6))
        self.assertTrue(model.mechanism_trace_["scenario_auxiliary_enabled"])
        self.assertTrue(model.mechanism_trace_["phase_p_ipcw_sample_reweighting"])
        self.assertTrue(model.mechanism_trace_["phase_p_model_selection"])
        self.assertTrue(model.mechanism_trace_["phase_p_platt_calibration"])
        self.assertIsNotNone(model.ensemble.phase_p_platt)
        metrics = model.evaluate(test)
        self.assertIn("macro_f1", metrics)
        report = model.counterfactual_report(test.head(1), row=0)
        self.assertIsInstance(report, list)
        _x, state = model.transform_state(test.head(1))
        forced = model.counterfactual._force_action(state, 1)
        derived = SocietyTransformer.action_derived_auxiliary_indices()
        self.assertTrue((forced[:, derived] == 0.0).all())
        guideline = SocietyTransformer.block_indices("guideline")
        self.assertEqual(forced[0, guideline.start], state[0, guideline.start])
        bad_draws = {
            action: np.zeros(model.counterfactual.bootstrap_replicates - 1)
            for action in range(6)
        }
        with self.assertRaisesRegex(ValueError, "exactly B="):
            model.counterfactual_report(
                test.head(1),
                row=0,
                patient_bootstrap_predictions=bad_draws,
                guideline_confidence_by_action={action: 1.0 for action in range(6)},
            )
        good_draws = {
            action: np.linspace(0.2, 0.8, model.counterfactual.bootstrap_replicates)
            + 0.01 * action
            for action in range(6)
        }
        cluster = model.clusterer.one_hot(_x) if model.clusterer is not None else None
        raw_sweep = model.counterfactual.sweep_patient(
            model,
            state[0],
            clinical_only=False,
            cluster_one_hot=cluster,
            patient_bootstrap_predictions=good_draws,
            guideline_confidence_by_action={action: 1.0 for action in range(6)},
        )
        self.assertTrue(
            all(
                item["bootstrap_replicates_used"]
                == model.counterfactual.bootstrap_replicates
                for item in raw_sweep
            )
        )
        phase_p = model.phase_p_observe(test.head(3))
        self.assertEqual(len(phase_p), 3)
        self.assertIn("action", phase_p[0])


if __name__ == "__main__":
    unittest.main()
