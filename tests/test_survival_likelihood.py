import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.config import apply_fast_overrides, load_config
from p_hlpl_hcc.data import generate_fixture_hcc_records
from p_hlpl_hcc.discrete_pipeline import DiscreteTimeSurvivalPipeline
from p_hlpl_hcc.survival import (
    discrete_time_negative_log_likelihood,
    fit_censoring_kaplan_meier,
    hazard_logits_to_class_probabilities,
    ipcw_likelihood_cell_weights,
    make_discrete_time_targets,
    risk_event_censor_counts,
)


class SurvivalLikelihoodTests(unittest.TestCase):
    def test_probabilities_are_ordered_class_distribution(self):
        logits = torch.zeros((3, 7), dtype=torch.float32)
        proba = hazard_logits_to_class_probabilities(logits)
        self.assertEqual(tuple(proba.shape), (3, 8))
        self.assertTrue(torch.allclose(proba.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.all(proba >= 0))

    def test_likelihood_ignores_unobserved_partial_interval(self):
        targets = make_discrete_time_targets([10.0], [0], [6.0, 12.0, 24.0])
        low_first_hazard = torch.tensor([[-4.0, 8.0, 8.0]])
        high_first_hazard = torch.tensor([[4.0, -8.0, -8.0]])
        y = torch.tensor(targets.event_targets)
        mask = torch.tensor(targets.likelihood_mask)
        low_loss = discrete_time_negative_log_likelihood(low_first_hazard, y, mask)
        high_loss = discrete_time_negative_log_likelihood(high_first_hazard, y, mask)
        self.assertLess(float(low_loss), float(high_loss))
        changed_unobserved = torch.tensor([[-4.0, -50.0, 50.0]])
        changed_loss = discrete_time_negative_log_likelihood(changed_unobserved, y, mask)
        self.assertAlmostEqual(float(low_loss), float(changed_loss), places=6)

    def test_ipcw_weights_vary_by_interval_and_event_time(self):
        times = np.array([4.0, 10.0, 20.0, 80.0])
        events = np.array([1, 0, 1, 0])
        cuts = [6.0, 12.0, 24.0]
        targets = make_discrete_time_targets(times, events, cuts)
        km = fit_censoring_kaplan_meier(times, events)
        weights = ipcw_likelihood_cell_weights(times, events, targets, km)
        self.assertEqual(weights.shape, targets.event_targets.shape)
        self.assertTrue(np.all(weights[targets.likelihood_mask] > 0))
        self.assertTrue(np.all(weights[~targets.likelihood_mask] == 0))
        # The later survival cell is up-weighted after censoring at month 10.
        self.assertGreater(weights[2, 1], weights[2, 0])
        logits = torch.zeros((4, 3), dtype=torch.float32)
        loss = discrete_time_negative_log_likelihood(
            logits,
            torch.tensor(targets.event_targets),
            torch.tensor(targets.likelihood_mask),
            cell_weights=torch.tensor(weights),
        )
        self.assertTrue(torch.isfinite(loss))

    def test_risk_counts_include_administratively_censored_tail(self):
        counts = risk_event_censor_counts([5, 12, 75, 90], [1, 1, 1, 0], [6, 12, 24, 72])
        self.assertEqual(len(counts), 5)
        self.assertEqual(counts[-1]["start_month"], 72.0)
        self.assertEqual(counts[-1]["administrative_censoring_at_horizon"], 2)
        self.assertEqual(counts[-1]["events"], 0)

    def test_discrete_pipeline_outputs_interval_hazards_and_ipcw_trace(self):
        config = apply_fast_overrides(load_config(ROOT / "configs" / "discrete_time.yaml"))
        config["phase_c"]["mlp"].update({"epochs": 1, "patience": 1, "hidden_dims": [8]})
        df = generate_fixture_hcc_records(n=48, seed=15)
        df.loc[0, ["overall_survival_months", "event"]] = [10.0, 0]
        model = DiscreteTimeSurvivalPipeline(config=config, seed=7).fit(df.iloc[:32], df.iloc[32:40])
        prepared = model._prepare(df.iloc[:32])
        self.assertTrue(prepared["survival_class"].isna().iloc[0])
        hazards = model.predict_hazards(df.iloc[40:])
        self.assertEqual(hazards.shape, (8, 7))
        self.assertTrue(np.all((hazards > 0) & (hazards < 1)))
        self.assertEqual(
            model.mechanism_trace_["ipcw_censoring_estimator"],
            "training_fold_reverse_kaplan_meier",
        )
        audited = model.evaluate(df.iloc[:32])
        self.assertEqual(audited["n_hard_label_auditable"], 31)
        self.assertEqual(audited["n_hard_label_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
