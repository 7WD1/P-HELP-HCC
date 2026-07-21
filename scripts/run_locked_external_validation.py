"""Evaluate a frozen Internal-673 pipeline without target-cohort refitting.

``strict_locked`` never fits on external rows.  ``anchor_recalibrated`` first
emits the strict result, then fits only a separate affine/Platt probability layer
on a disjoint, explicitly supplied anchor set of at most 30 patients.  The latter
is a target-adapted sensitivity analysis and is never labeled strict validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.constants import N_CLASSES
from p_hlpl_hcc.data import load_table, normalize_label_series, survival_months_to_class
from p_hlpl_hcc.locked_validation import (
    AnchorAffinePlattCalibrator,
    hash_contract,
    model_contract_digests,
    sha256_file,
    verify_freeze_manifest,
)
from p_hlpl_hcc.metrics import classification_metrics
from p_hlpl_hcc.preprocessing import EXTERNAL_MISSINGNESS_FEATURES
from p_hlpl_hcc.utils import ensure_dir, save_json


def load_freeze_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Freeze manifest must contain a JSON object")
    return manifest


def _endpoint_arrays(df: pd.DataFrame, data_cfg: dict) -> dict[str, np.ndarray | pd.Series]:
    time_col = str(data_cfg["target_time_col"])
    event_col = str(data_cfg["event_col"])
    label_col = str(data_cfg["label_col"])
    for column in (time_col, event_col):
        if column not in df:
            raise ValueError(f"External cohort is missing required column: {column}")
    time = pd.to_numeric(df[time_col], errors="raise").to_numpy(float)
    event = pd.to_numeric(df[event_col], errors="raise").to_numpy(int)
    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError(f"External event column {event_col} must contain only 0/1")
    labels = (
        normalize_label_series(df[label_col])
        if label_col in df
        else pd.Series(survival_months_to_class(time), index=df.index)
    )
    horizon = float(data_cfg.get("survival_bins_months", [0, 72])[-1])
    hard_endpoint_observed = (event == 1) | (time >= horizon)
    return {
        "time": time,
        "event": event,
        "labels": labels,
        "hard_endpoint_observed": hard_endpoint_observed,
    }


def _inference_frame(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    return df.drop(
        columns=[
            data_cfg["target_time_col"],
            data_cfg["event_col"],
            data_cfg["label_col"],
        ],
        errors="ignore",
    )


def _predict_frozen(
    model: object, df: pd.DataFrame, data_cfg: dict
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predict without fitting and return the pre-imputation missingness mask."""

    inference_df = _inference_frame(df, data_cfg)
    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is None or not hasattr(preprocessor, "missingness_indicators"):
        raise ValueError("Frozen model lacks the missingness-indicator contract")
    missingness = preprocessor.missingness_indicators(inference_df)
    probabilities = np.asarray(model.predict_proba(inference_df), dtype=float)
    if probabilities.shape != (len(df), N_CLASSES):
        raise ValueError(
            f"Expected {N_CLASSES}-class probabilities, got {probabilities.shape}"
        )
    return probabilities, missingness


def _missingness_summary(missingness: pd.DataFrame) -> dict:
    counts = {name: int(value) for name, value in missingness.sum(axis=0).items()}
    rates = {name: float(value) for name, value in missingness.mean(axis=0).items()}
    focus_names = [f"{name}__missing" for name in EXTERNAL_MISSINGNESS_FEATURES]
    return {
        "indicator_contract": "pre-imputation sidecar; not appended to frozen 67-feature matrix",
        "counts": counts,
        "rates": rates,
        "externally_audited": {
            name: {"count": counts[name], "rate": rates[name]}
            for name in focus_names
            if name in counts
        },
    }


def _metrics(
    probabilities: np.ndarray,
    endpoints: dict[str, np.ndarray | pd.Series],
    *,
    cohort: str,
    mode: str,
) -> dict:
    observed = np.asarray(endpoints["hard_endpoint_observed"], dtype=bool)
    labels = np.asarray(endpoints["labels"], dtype=int)
    time = np.asarray(endpoints["time"], dtype=float)
    event = np.asarray(endpoints["event"], dtype=int)
    result = classification_metrics(
        labels[observed],
        probabilities[observed],
        times=time[observed],
        events=event[observed],
    )
    result.update(
        {
            "cohort": cohort,
            "evaluation_mode": mode,
            "locked_base_model": True,
            "strict_locked_result": mode == "strict_locked",
            "base_model_retraining_performed": False,
            "target_label_fit_performed": mode != "strict_locked",
            "recalibration_performed": mode != "strict_locked",
            "n_rows": int(len(labels)),
            "n_hard_endpoint_rows": int(observed.sum()),
            "n_immature_censored_excluded_from_class_metrics": int((~observed).sum()),
        }
    )
    return result


def _prediction_table(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    endpoints: dict[str, np.ndarray | pd.Series],
    missingness: pd.DataFrame,
) -> pd.DataFrame:
    observed = np.asarray(endpoints["hard_endpoint_observed"], dtype=bool)
    labels = pd.Series(endpoints["labels"], index=df.index, dtype="Int64")
    labels.loc[~observed] = pd.NA
    table = pd.DataFrame(
        {
            "patient_id": df["patient_id"].astype(str).to_numpy()
            if "patient_id" in df
            else [f"row_{index}" for index in range(len(df))],
            "true_class": labels,
            "event": np.asarray(endpoints["event"], dtype=int),
            "overall_survival_months": np.asarray(endpoints["time"], dtype=float),
            "hard_endpoint_observed": observed,
            "pred_class": np.argmax(probabilities, axis=1),
        }
    )
    for class_index in range(N_CLASSES):
        table[f"p_c{class_index + 1}"] = probabilities[:, class_index]
    for feature in EXTERNAL_MISSINGNESS_FEATURES:
        column = f"{feature}__missing"
        if column in missingness:
            table[column] = missingness[column].to_numpy(dtype=np.uint8)
    return table


def _assert_disjoint_anchor(anchor_df: pd.DataFrame, evaluation_df: pd.DataFrame) -> None:
    if "patient_id" not in anchor_df or "patient_id" not in evaluation_df:
        raise ValueError(
            "Anchor and evaluation tables must both contain patient_id so disjointness can be audited"
        )
    anchor_ids = set(anchor_df["patient_id"].astype(str))
    evaluation_ids = set(evaluation_df["patient_id"].astype(str))
    overlap = sorted(anchor_ids.intersection(evaluation_ids))
    if overlap:
        preview = overlap[:5]
        raise ValueError(f"Anchor and evaluation patients overlap: {preview}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locked-model external validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("strict_locked", "anchor_recalibrated"),
        default="strict_locked",
        help="Strict mode performs no target fit; anchor mode is a separate sensitivity analysis.",
    )
    parser.add_argument(
        "--anchor-data",
        default=None,
        help="Disjoint labeled target-cohort anchor table, required only for anchor_recalibrated.",
    )
    parser.add_argument("--anchor-max-patients", type=int, default=30)
    parser.add_argument("--anchor-seed", type=int, default=42)
    parser.add_argument(
        "--trusted-model",
        action="store_true",
        help="Required acknowledgement: joblib files can execute code and must be trusted.",
    )
    args = parser.parse_args(argv)
    if not args.trusted_model:
        raise ValueError("Refusing to load joblib without --trusted-model")
    if args.mode == "strict_locked" and args.anchor_data is not None:
        raise ValueError("strict_locked forbids --anchor-data and all target-cohort fitting")
    if args.mode == "anchor_recalibrated" and args.anchor_data is None:
        raise ValueError("anchor_recalibrated requires --anchor-data")
    if not 1 <= int(args.anchor_max_patients) <= 30:
        raise ValueError("--anchor-max-patients must lie in 1..30")

    model_path = Path(args.model)
    model = joblib.load(model_path)
    manifest = load_freeze_manifest(Path(args.freeze_manifest))
    verified_hashes = verify_freeze_manifest(manifest, model, model_path)
    contracts_before = model_contract_digests(model)
    in_memory_state_before = joblib.hash(model, hash_name="sha1")

    evaluation_df = load_table(args.data)
    data_cfg = model.config["data"]
    # Produce frozen probabilities from predictors first.  Outcome columns are
    # dropped inside _predict_frozen and are read only afterwards for scoring.
    strict_probabilities, missingness = _predict_frozen(model, evaluation_df, data_cfg)
    endpoints = _endpoint_arrays(evaluation_df, data_cfg)
    strict_metrics = _metrics(
        strict_probabilities,
        endpoints,
        cohort=args.cohort,
        mode="strict_locked",
    )
    strict_metrics["freeze_contract_verified"] = True
    strict_metrics["missingness"] = _missingness_summary(missingness)

    output = ensure_dir(args.output_dir)
    strict_prediction_path = output / "locked_external_strict_predictions.csv"
    _prediction_table(
        evaluation_df, strict_probabilities, endpoints, missingness
    ).to_csv(strict_prediction_path, index=False)
    strict_metrics_path = output / "locked_external_strict_metrics.json"
    save_json(strict_metrics, strict_metrics_path)

    output_hashes: dict[str, object] = {
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        **verified_hashes,
        "external_data_sha256": sha256_file(args.data),
        "strict_locked_predictions_sha256": sha256_file(strict_prediction_path),
        "strict_locked_metrics_sha256": sha256_file(strict_metrics_path),
    }

    if args.mode == "anchor_recalibrated":
        anchor_df = load_table(args.anchor_data)
        _assert_disjoint_anchor(anchor_df, evaluation_df)
        anchor_endpoints = _endpoint_arrays(anchor_df, data_cfg)
        anchor_observed = np.asarray(anchor_endpoints["hard_endpoint_observed"], dtype=bool)
        if not anchor_observed.all():
            raise ValueError(
                "Anchor data contain immature censored rows; every anchor label must be observed"
            )
        if len(anchor_df) > int(args.anchor_max_patients):
            raise ValueError(
                f"Anchor table has {len(anchor_df)} rows; allowed maximum is "
                f"{args.anchor_max_patients}"
            )
        anchor_probabilities, anchor_missingness = _predict_frozen(model, anchor_df, data_cfg)
        anchor_labels = np.asarray(anchor_endpoints["labels"], dtype=int)
        recalibrator = AnchorAffinePlattCalibrator(
            max_anchor_patients=int(args.anchor_max_patients),
            random_state=int(args.anchor_seed),
        ).fit(anchor_probabilities, anchor_labels)
        adapted_probabilities = recalibrator.predict(strict_probabilities)
        adapted_metrics = _metrics(
            adapted_probabilities,
            endpoints,
            cohort=args.cohort,
            mode="anchor_recalibrated",
        )
        adapted_metrics.update(
            {
                "freeze_contract_verified": True,
                "anchor_data_disjoint_from_evaluation": True,
                "n_anchor_patients": int(len(anchor_df)),
                "anchor_recalibration_contract_sha256": hash_contract(
                    recalibrator.contract()
                ),
                "missingness": _missingness_summary(missingness),
                "anchor_missingness": _missingness_summary(anchor_missingness),
            }
        )
        adapted_prediction_path = output / "locked_external_anchor_recalibrated_predictions.csv"
        _prediction_table(
            evaluation_df, adapted_probabilities, endpoints, missingness
        ).to_csv(adapted_prediction_path, index=False)
        adapted_metrics_path = output / "locked_external_anchor_recalibrated_metrics.json"
        save_json(adapted_metrics, adapted_metrics_path)
        recalibration_contract_path = output / "anchor_recalibration_contract.json"
        save_json(recalibrator.contract(), recalibration_contract_path)
        output_hashes.update(
            {
                "anchor_data_sha256": sha256_file(args.anchor_data),
                "anchor_recalibrated_predictions_sha256": sha256_file(
                    adapted_prediction_path
                ),
                "anchor_recalibrated_metrics_sha256": sha256_file(adapted_metrics_path),
                "anchor_recalibration_contract_file_sha256": sha256_file(
                    recalibration_contract_path
                ),
            }
        )

    contracts_after = model_contract_digests(model)
    if contracts_after != contracts_before:
        raise RuntimeError("Frozen model contract changed during external validation")
    in_memory_state_after = joblib.hash(model, hash_name="sha1")
    if in_memory_state_after != in_memory_state_before:
        raise RuntimeError("Frozen model state changed during external validation")
    output_hashes["frozen_contract_unchanged_after_evaluation"] = True
    output_hashes["frozen_model_state_unchanged_after_evaluation"] = True
    output_hashes["in_memory_model_state_sha1"] = in_memory_state_after
    save_json(output_hashes, output / "locked_external_hashes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
