import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_help_hcc.config import apply_fast_overrides, load_config
from p_help_hcc.data import generate_synthetic_hcc_cohort, validate_and_prepare_dataframe
from p_help_hcc.pipeline import PHelpHCCPipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_fit_predict_counterfactual(self):
        config = apply_fast_overrides(load_config(ROOT / "configs" / "default.yaml"))
        df = validate_and_prepare_dataframe(generate_synthetic_hcc_cohort(n=72, seed=11))
        train = df.iloc[:48].reset_index(drop=True)
        val = df.iloc[48:60].reset_index(drop=True)
        test = df.iloc[60:].reset_index(drop=True)
        model = PHelpHCCPipeline(config=config, seed=42).fit(train, val)
        proba = model.predict_proba(test)
        self.assertEqual(proba.shape, (len(test), 8))
        self.assertTrue(((proba.sum(axis=1) - 1.0) ** 2).max() < 1e-8)
        inference_only = test.drop(columns=["overall_survival_months", "event", "survival_class"])
        infer_proba = model.predict_proba(inference_only)
        self.assertEqual(infer_proba.shape, (len(test), 8))
        metrics = model.evaluate(test)
        self.assertIn("macro_f1", metrics)
        report = model.counterfactual_report(test.head(1), row=0)
        self.assertIsInstance(report, list)
        phase_p = model.phase_p_observe(test.head(3))
        self.assertEqual(len(phase_p), 3)
        self.assertIn("action", phase_p[0])


if __name__ == "__main__":
    unittest.main()
