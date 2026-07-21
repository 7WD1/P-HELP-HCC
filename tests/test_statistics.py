import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.statistics import (
    decision_curve_net_benefit,
    e_value_from_risk_ratio,
    mcnemar_exact,
    nadeau_bengio_corrected_ttest,
    paired_ipcw_c_index_test,
    patient_bootstrap_interval,
)


class StatisticalAuditTests(unittest.TestCase):
    def test_patient_bootstrap_is_deterministic(self):
        values = np.arange(12, dtype=float)
        first = patient_bootstrap_interval(values, replicates=50, random_state=8)
        second = patient_bootstrap_interval(values, replicates=50, random_state=8)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_unit"], "patient")

    def test_paired_tests_and_decision_curve(self):
        corrected = nadeau_bengio_corrected_ttest(
            np.array([0.03, 0.02, 0.01, 0.04, 0.02]), test_fraction=0.2
        )
        self.assertGreater(corrected["correction_factor"], 1 / 5)
        mcnemar = mcnemar_exact(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]), np.array([0, 0, 0, 1])
        )
        self.assertEqual(mcnemar["discordant"], 3)
        curve = decision_curve_net_benefit(
            np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.7, 0.4]), np.array([0.2])
        )
        self.assertEqual(len(curve), 1)
        self.assertGreater(e_value_from_risk_ratio(1.5), 1.5)

    def test_paired_ipcw_c_index_variance_path(self):
        times = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        events = np.array([1, 1, 0, 1, 0, 1], dtype=int)
        model_risk = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        comparator_risk = model_risk[::-1]
        first = paired_ipcw_c_index_test(
            times,
            events,
            model_risk,
            comparator_risk,
            replicates=100,
            random_state=9,
        )
        second = paired_ipcw_c_index_test(
            times,
            events,
            model_risk,
            comparator_risk,
            replicates=100,
            random_state=9,
        )
        self.assertEqual(first, second)
        self.assertGreater(first["model_ipcw_c_index"], first["comparator_ipcw_c_index"])
        self.assertEqual(first["bootstrap_unit"], "patient")


if __name__ == "__main__":
    unittest.main()
