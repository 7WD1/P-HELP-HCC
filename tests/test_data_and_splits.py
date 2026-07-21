import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.data import generate_fixture_hcc_records, survival_months_to_class, validate_and_prepare_dataframe
from p_hlpl_hcc.splits import build_repeated_splits
from p_hlpl_hcc.survival import make_discrete_time_targets, risk_event_censor_counts


class DataAndSplitTests(unittest.TestCase):
    def test_survival_class_boundaries(self):
        labels = survival_months_to_class([0.5, 6, 11.9, 12, 24, 72, 90])
        self.assertEqual(labels.tolist(), [0, 1, 1, 2, 3, 7, 7])

    def test_fixture_data_schema_and_splits(self):
        df = generate_fixture_hcc_records(n=80, seed=7)
        prepared = validate_and_prepare_dataframe(df)
        self.assertEqual(len(prepared), 80)
        self.assertIn("survival_class", prepared.columns)
        splits = build_repeated_splits(prepared, seeds=[42], outer_folds=2)
        folds = splits["42"]["folds"]
        self.assertEqual(len(folds), 2)
        all_idx = set(range(len(prepared)))
        for fold in folds:
            train = set(fold["train"])
            val = set(fold["val"])
            test = set(fold["test"])
            self.assertTrue(train.isdisjoint(val))
            self.assertTrue(train.isdisjoint(test))
            self.assertTrue(val.isdisjoint(test))
            self.assertEqual(train | val | test, all_idx)

    def test_hard_labels_reject_immature_censoring_even_when_supplied(self):
        df = generate_fixture_hcc_records(n=8, seed=4)
        df.loc[0, ["overall_survival_months", "event", "survival_class"]] = [10.0, 0, 1]
        with self.assertRaisesRegex(ValueError, "definitive eight-class endpoint is undefined"):
            validate_and_prepare_dataframe(df)
        prepared = validate_and_prepare_dataframe(df, require_unambiguous_hard_labels=False)
        self.assertEqual(len(prepared), len(df))

    def test_discrete_time_targets_mask_partial_censoring(self):
        targets = make_discrete_time_targets(
            [4.0, 10.0, 12.0, 80.0],
            [1, 0, 1, 0],
            [6.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
        )
        self.assertEqual(targets.event_interval.tolist(), [0, -1, 2, -1])
        self.assertEqual(targets.likelihood_mask[1].tolist(), [True, False, False, False, False, False, False])
        self.assertEqual(targets.likelihood_mask[3].sum(), 7)
        counts = risk_event_censor_counts(
            [4.0, 10.0, 12.0, 80.0], [1, 0, 1, 0], [6.0, 12.0, 24.0]
        )
        self.assertEqual(counts[0]["risk_set"], 4)
        self.assertEqual(counts[0]["events"], 1)


if __name__ == "__main__":
    unittest.main()

