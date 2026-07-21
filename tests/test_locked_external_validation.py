import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p_hlpl_hcc.data import generate_fixture_hcc_records
from p_hlpl_hcc.locked_validation import (
    AnchorAffinePlattCalibrator,
    build_freeze_manifest,
    verify_freeze_manifest,
)
from p_hlpl_hcc.preprocessing import PHlplPreprocessor
from run_locked_external_validation import _predict_frozen, main as locked_main


class FrozenToyModel:
    """Minimal serialized pipeline exposing the locked inference contracts."""

    def __init__(self, train_df):
        self.config = {
            "data": {
                "target_time_col": "overall_survival_months",
                "event_col": "event",
                "label_col": "survival_class",
                "survival_bins_months": [0, 6, 12, 24, 36, 48, 60, 72],
            },
            "phase_p": {
                "soft_error_threshold": 0.18,
                "hard_error_threshold": 0.32,
                "abstention_entropy_soft": 0.65,
                "abstention_entropy_hard": 0.85,
            },
        }
        self.preprocessor = PHlplPreprocessor().fit(train_df)
        self.ensemble = SimpleNamespace(
            models={},
            active_model_names_=[],
            fusion_weights=np.asarray([1.0]),
            calibration_head=None,
            phase_p_platt=None,
        )
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("Target-cohort fit must never be called")

    def predict_proba(self, df):
        x = self.preprocessor.transform(df)
        logits = np.column_stack([x[:, 0] * (index + 1) / 20 for index in range(8)])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class LockedExternalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = generate_fixture_hcc_records(n=40, seed=121)

    def test_missingness_sidecar_preserves_frozen_67_columns(self):
        preprocessor = PHlplPreprocessor().fit(self.train)
        external = self.train.head(5).drop(columns=["afp", "child_pugh_b_or_c"])
        transformed, indicators = preprocessor.transform_with_missingness(external)
        self.assertEqual(transformed.shape, (5, 67))
        self.assertTrue((indicators["log_afp__missing"] == 1).all())
        self.assertTrue((indicators["child_pugh_b_or_c__missing"] == 1).all())
        self.assertTrue((indicators["afp_gt400__missing"] == 1).all())
        self.assertTrue((indicators["resection_eligible__missing"] == 1).all())
        self.assertEqual(preprocessor.imputation_contract()["strategy"], "median")
        self.assertFalse(
            preprocessor.preprocessing_contract()["missingness_sidecar"][
                "concatenated_to_model_input"
            ]
        )

    def test_freeze_manifest_verifies_each_contract_and_detects_tampering(self):
        model = FrozenToyModel(self.train)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            joblib.dump(model, model_path)
            manifest = build_freeze_manifest(model, model_path)
            verified = verify_freeze_manifest(manifest, model, model_path)
            self.assertIn("calibration_contract_sha256", verified)
            tampered = dict(manifest)
            tampered["imputation_contract_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "imputation_contract"):
                verify_freeze_manifest(tampered, model, model_path)

    def test_strict_prediction_uses_transform_only_and_carries_seer_markers(self):
        model = FrozenToyModel(self.train)
        external = self.train.head(7).drop(columns=["afp", "child_pugh_b_or_c"])
        data_cfg = model.config["data"]
        inference = external.drop(
            columns=["overall_survival_months", "event", "survival_class"]
        )
        probabilities, indicators = _predict_frozen(model, inference, data_cfg)
        self.assertEqual(probabilities.shape, (7, 8))
        self.assertEqual(model.fit_calls, 0)
        self.assertTrue((indicators["log_afp__missing"] == 1).all())
        self.assertTrue((indicators["child_pugh_b_or_c__missing"] == 1).all())

    def test_strict_cli_rejects_anchor_input_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "strict_locked forbids"):
            locked_main(
                [
                    "--model",
                    "unused.joblib",
                    "--freeze-manifest",
                    "unused.json",
                    "--data",
                    "unused.csv",
                    "--cohort",
                    "SEER",
                    "--output-dir",
                    "unused-output",
                    "--anchor-data",
                    "forbidden.csv",
                    "--trusted-model",
                ]
            )

    def test_strict_cli_emits_verified_missingness_audit_without_fit(self):
        model = FrozenToyModel(self.train)
        evaluation = self.train.head(16).copy()
        evaluation["patient_id"] = [f"EXT{index:03d}" for index in range(len(evaluation))]
        evaluation["event"] = 1
        evaluation["survival_class"] = np.tile(np.arange(8), 2)
        evaluation["overall_survival_months"] = np.tile(
            np.asarray([3, 9, 18, 30, 42, 54, 66, 80], dtype=float), 2
        )
        evaluation = evaluation.drop(columns=["afp", "child_pugh_b_or_c"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.joblib"
            manifest_path = root / "freeze.json"
            data_path = root / "external.csv"
            output = root / "out"
            joblib.dump(model, model_path)
            manifest_path.write_text(
                json.dumps(build_freeze_manifest(model, model_path)), encoding="utf-8"
            )
            evaluation.to_csv(data_path, index=False)
            result = locked_main(
                [
                    "--model",
                    str(model_path),
                    "--freeze-manifest",
                    str(manifest_path),
                    "--data",
                    str(data_path),
                    "--cohort",
                    "SEER",
                    "--output-dir",
                    str(output),
                    "--trusted-model",
                ]
            )
            self.assertEqual(result, 0)
            metrics = json.loads(
                (output / "locked_external_strict_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(metrics["strict_locked_result"])
            self.assertFalse(metrics["target_label_fit_performed"])
            self.assertEqual(
                metrics["missingness"]["externally_audited"]["log_afp__missing"][
                    "count"
                ],
                len(evaluation),
            )
            hashes = json.loads(
                (output / "locked_external_hashes.json").read_text(encoding="utf-8")
            )
            self.assertTrue(hashes["frozen_contract_unchanged_after_evaluation"])

    def test_anchor_affine_platt_is_separate_and_bounded(self):
        rng = np.random.default_rng(91)
        raw = rng.uniform(0.01, 1.0, size=(24, 8))
        probabilities = raw / raw.sum(axis=1, keepdims=True)
        labels = np.tile(np.arange(8), 3)
        calibrator = AnchorAffinePlattCalibrator(max_anchor_patients=30).fit(
            probabilities, labels
        )
        adapted = calibrator.predict(probabilities[:4])
        self.assertEqual(adapted.shape, (4, 8))
        np.testing.assert_allclose(adapted.sum(axis=1), 1.0)
        contract = calibrator.contract()
        self.assertFalse(contract["strict_locked_result"])
        self.assertTrue(contract["target_adaptation"])
        with self.assertRaisesRegex(ValueError, "maximum"):
            AnchorAffinePlattCalibrator(max_anchor_patients=20).fit(
                probabilities, labels
            )

    def test_schema_requires_all_frozen_components(self):
        schema = json.loads(
            (ROOT / "schemas" / "locked_external_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        required = set(schema["required"])
        self.assertTrue(
            {
                "preprocessing_contract_sha256",
                "imputation_contract_sha256",
                "calibration_contract_sha256",
                "decision_contract_sha256",
            }.issubset(required)
        )


if __name__ == "__main__":
    unittest.main()
