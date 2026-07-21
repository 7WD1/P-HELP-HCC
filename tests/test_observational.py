import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.observational import (
    CrossFittedDoublyRobust,
    binary_survival_outcome,
    iptw_rmst_contrasts,
    validate_pretreatment_feature_names,
)


class ObservationalAnalysisTests(unittest.TestCase):
    def test_horizon_outcome_excludes_immature_censoring(self):
        outcome, observed = binary_survival_outcome([5, 8, 12, 20], [1, 0, 0, 0], 12)
        self.assertEqual(observed.tolist(), [True, False, True, True])
        self.assertEqual(outcome.tolist(), [False, False, False, True])

    def test_assignment_encoding_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pretreatment covariates only"):
            validate_pretreatment_feature_names(["age", "treatment_resection"])

    def test_cross_fitted_dr_is_deterministic_and_reports_balance(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(120, 3))
        propensity = 1.0 / (1.0 + np.exp(-0.4 * x[:, 0]))
        treatment = (rng.random(120) < propensity).astype(int)
        outcome_p = 1.0 / (1.0 + np.exp(-(-0.2 + 0.3 * x[:, 0] + 0.25 * treatment)))
        outcome = (rng.random(120) < outcome_p).astype(float)
        kwargs = dict(
            n_splits=3,
            trim=0.02,
            bootstrap_replicates=50,
            random_state=9,
        )
        first = CrossFittedDoublyRobust(**kwargs).fit(
            x,
            treatment,
            outcome,
            feature_names=["age", "tumor_size", "albumin"],
            reference_action=0,
        ).report()
        second = CrossFittedDoublyRobust(**kwargs).fit(
            x,
            treatment,
            outcome,
            feature_names=["age", "tumor_size", "albumin"],
            reference_action=0,
        ).report()
        self.assertEqual(first, second)
        contrast = first["contrasts"][0]
        self.assertGreater(contrast["on_support_n"], 0)
        self.assertIn("max_abs_smd_after_iptw", contrast)
        self.assertEqual(set(contrast["estimators"]), {"naive", "iptw", "aipw_dr"})
        self.assertIn("e_value_point", contrast)
        self.assertIn("retention_by_observed_arm", contrast)
        self.assertEqual(contrast["bootstrap_replicates"], 50)
        self.assertFalse(first["causal_claim"])

    def test_iptw_km_rmst_bootstrap_refits_propensity(self):
        rng = np.random.default_rng(30)
        x = rng.normal(size=(80, 2))
        treatment = (rng.random(80) < 0.5).astype(int)
        times = np.clip(rng.exponential(18, size=80) + 1, 1, 36)
        events = (rng.random(80) < 0.75).astype(int)
        result = iptw_rmst_contrasts(
            x,
            treatment,
            times,
            events,
            reference_action=0,
            tau=12.0,
            trim=0.02,
            n_splits=3,
            bootstrap_replicates=10,
            random_state=5,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["bootstrap_refits_propensity_model"])
        self.assertEqual(result[0]["bootstrap_replicates_requested"], 10)
        self.assertIn("iptw_km_rmst_difference_months", result[0])


if __name__ == "__main__":
    unittest.main()
