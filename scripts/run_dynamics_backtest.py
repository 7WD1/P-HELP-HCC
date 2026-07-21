"""Back-test physical-unit tumor/fibrosis rules on longitudinal observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.config import apply_named_variant, load_config
from p_hlpl_hcc.data import load_table
from p_hlpl_hcc.statistics import patient_bootstrap_interval
from p_hlpl_hcc.utils import ensure_dir, save_json


REQUIRED_COLUMNS = {
    "patient_id",
    "start_month",
    "end_month",
    "tumor_diameter_start_cm",
    "tumor_diameter_observed_cm",
    "growth_rate_per_month",
    "treatment_shrinkage_rate_per_month",
    "fibrosis_start",
    "fibrosis_observed",
    "kappa_age_per_month",
    "kappa_treatment_per_month",
    "kappa_recovery_per_month",
    "recovery_score",
    "loco_regional_action",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Longitudinal physical-unit dynamics back-test")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", required=True, choices=["static_only", "gompertz_only", "fibrosis_only", "full_dynamics"])
    parser.add_argument("--variants-config", default="configs/dynamics.yaml")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = apply_named_variant(
        load_config(args.config),
        args.variants_config,
        args.variant,
        experiment_key="dynamics_variant",
    )
    df = load_table(args.data)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Longitudinal back-test table is missing columns: {missing}")
    if df["patient_id"].nunique() < 2:
        raise ValueError("Patient-level uncertainty requires at least two distinct patients")
    numeric_columns = sorted(REQUIRED_COLUMNS.difference({"patient_id"}))
    values = df[numeric_columns].apply(pd.to_numeric, errors="raise")
    dt = values["end_month"].to_numpy() - values["start_month"].to_numpy()
    if np.any(dt <= 0):
        raise ValueError("Every longitudinal interval must have end_month > start_month")
    phase_a = config["phase_a"]
    tumor_enabled = bool(phase_a.get("tumor_update_enabled", True))
    fibrosis_enabled = bool(phase_a.get("fibrosis_update_enabled", True))
    diameter = values["tumor_diameter_start_cm"].to_numpy(float)
    if np.any(diameter <= 0):
        raise ValueError("Tumor diameters must be positive physical centimeter values")
    d_max = float(phase_a.get("d_max_cm", 20.0))
    if tumor_enabled:
        growth = values["growth_rate_per_month"].to_numpy(float)
        shrinkage = values["treatment_shrinkage_rate_per_month"].to_numpy(float)
        diameter_pred = diameter * np.exp(growth * np.log(d_max / diameter) * dt)
        diameter_pred -= shrinkage * diameter * dt
        diameter_pred = np.clip(diameter_pred, 0.0, d_max)
    else:
        diameter_pred = diameter.copy()
    fibrosis = values["fibrosis_start"].to_numpy(float)
    if fibrosis_enabled:
        fibrosis_pred = fibrosis + values["kappa_age_per_month"].to_numpy(float) * dt
        fibrosis_pred += (
            values["kappa_treatment_per_month"].to_numpy(float)
            * values["loco_regional_action"].to_numpy(float)
            * dt
        )
        fibrosis_pred -= (
            values["kappa_recovery_per_month"].to_numpy(float)
            * values["recovery_score"].to_numpy(float)
            * dt
        )
        fibrosis_pred = np.clip(fibrosis_pred, 0.0, 1.0)
    else:
        fibrosis_pred = fibrosis.copy()
    diameter_error = diameter_pred - values["tumor_diameter_observed_cm"].to_numpy(float)
    fibrosis_error = fibrosis_pred - values["fibrosis_observed"].to_numpy(float)
    output = ensure_dir(args.output_dir)
    table = df.copy()
    table["tumor_diameter_predicted_cm"] = diameter_pred
    table["tumor_diameter_error_cm"] = diameter_error
    table["fibrosis_predicted"] = fibrosis_pred
    table["fibrosis_error"] = fibrosis_error
    table.to_csv(output / "longitudinal_predictions.csv", index=False)
    patient_tumor_mae = (
        table.assign(_abs_tumor_error=np.abs(diameter_error))
        .groupby("patient_id", sort=True)["_abs_tumor_error"]
        .mean()
        .to_numpy(float)
    )
    summary = {
        "variant": args.variant,
        "physical_units_validated_by_schema": True,
        "tumor_update_enabled": tumor_enabled,
        "fibrosis_update_enabled": fibrosis_enabled,
        "n_intervals": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "affects_main_classifier_predict_path": False,
        "parameter_uncertainty_propagated": False,
        "parameter_uncertainty_note": (
            "The input supplies patient/interval parameters without fitted parameter "
            "distributions; this path reports patient-level error uncertainty only."
        ),
        "tumor_bias_cm": float(np.mean(diameter_error)),
        "tumor_mae_cm": float(np.mean(np.abs(diameter_error))),
        "tumor_limits_of_agreement_cm": [
            float(np.mean(diameter_error) - 1.96 * np.std(diameter_error, ddof=1)),
            float(np.mean(diameter_error) + 1.96 * np.std(diameter_error, ddof=1)),
        ],
        "fibrosis_bias": float(np.mean(fibrosis_error)),
        "fibrosis_mae": float(np.mean(np.abs(fibrosis_error))),
        "tumor_mae_patient_bootstrap": patient_bootstrap_interval(
            patient_tumor_mae, replicates=args.bootstrap, random_state=args.seed
        ),
    }
    save_json(summary, output / "longitudinal_summary.json")
    save_json(config, output / "resolved_dynamics_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
