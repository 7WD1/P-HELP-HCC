import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.cox import (
    CoxElasticNetTorch,
    breslow_baseline_hazard,
    breslow_negative_partial_log_likelihood,
)


class CoxBreslowTests(unittest.TestCase):
    def test_partial_likelihood_groups_tied_deaths(self):
        log_risk = torch.tensor(
            [math.log(2.0), math.log(3.0), math.log(5.0)],
            dtype=torch.float64,
        )
        loss = breslow_negative_partial_log_likelihood(
            log_risk,
            np.array([5.0, 5.0, 4.0]),
            np.array([1, 1, 1]),
        )
        partial = (
            math.log(2.0)
            + math.log(3.0)
            - 2.0 * math.log(5.0)
            + math.log(5.0)
            - math.log(10.0)
        )
        self.assertAlmostEqual(float(loss), -partial, places=12)

    def test_fit_stores_breslow_baseline_hazard(self):
        model = CoxElasticNetTorch(epochs=0, l1=0.0, l2=0.0).fit(
            np.zeros((3, 1), dtype=float),
            np.array([5.0, 5.0, 4.0]),
            np.array([1, 1, 1]),
        )
        np.testing.assert_allclose(model.baseline_event_times_, [4.0, 5.0])
        np.testing.assert_allclose(model.baseline_hazard_increments_, [1.0 / 3.0, 1.0])
        np.testing.assert_allclose(
            model.baseline_cumulative_hazard_at([3.0, 4.0, 5.0]),
            [0.0, 1.0 / 3.0, 4.0 / 3.0],
        )

    def test_baseline_denominator_uses_unclipped_log_risk(self):
        event_times, increments, cumulative = breslow_baseline_hazard(
            np.array([80.0, 0.0], dtype=float),
            np.array([5.0, 4.0], dtype=float),
            np.array([1, 1], dtype=int),
        )
        expected_first = math.exp(-80.0) / (1.0 + math.exp(-80.0))
        np.testing.assert_allclose(event_times, [4.0, 5.0])
        self.assertAlmostEqual(increments[0], expected_first, places=48)
        self.assertAlmostEqual(increments[1], math.exp(-80.0), places=48)
        np.testing.assert_allclose(cumulative, np.cumsum(increments))


if __name__ == "__main__":
    unittest.main()
