import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_help_hcc.data import generate_synthetic_hcc_cohort, survival_months_to_class, validate_and_prepare_dataframe
from p_help_hcc.splits import build_repeated_splits


class DataAndSplitTests(unittest.TestCase):
    def test_survival_class_boundaries(self):
        labels = survival_months_to_class([0.5, 6, 11.9, 12, 24, 72, 90])
        self.assertEqual(labels.tolist(), [0, 1, 1, 2, 3, 7, 7])

    def test_synthetic_data_schema_and_splits(self):
        df = generate_synthetic_hcc_cohort(n=80, seed=7)
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


if __name__ == "__main__":
    unittest.main()

