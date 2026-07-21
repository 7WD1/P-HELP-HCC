"""Run a pretreatment-only cross-fitted observational sensitivity analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.data import load_table
from p_hlpl_hcc.observational import (
    CrossFittedDoublyRobust,
    binary_survival_outcome,
    iptw_rmst_contrasts,
)
from p_hlpl_hcc.utils import save_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-fitted doubly robust observational scenario-sensitivity analysis"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--covariates", nargs="+", required=True, help="Explicit pretreatment columns")
    parser.add_argument("--treatment-col", required=True)
    parser.add_argument("--time-col", default="overall_survival_months")
    parser.add_argument("--event-col", default="event")
    parser.add_argument("--horizon", type=float, default=12.0)
    parser.add_argument("--reference-action", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trim", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    df = load_table(args.data)
    required = set(args.covariates + [args.treatment_col, args.time_col, args.event_col])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    numeric = df[args.covariates].apply(pd.to_numeric, errors="raise")
    outcome, observed = binary_survival_outcome(df[args.time_col], df[args.event_col], args.horizon)
    model = CrossFittedDoublyRobust(
        n_splits=args.folds,
        trim=args.trim,
        bootstrap_replicates=args.bootstrap,
        random_state=args.seed,
    ).fit(
        numeric.to_numpy()[observed],
        pd.to_numeric(df.loc[observed, args.treatment_col], errors="raise").to_numpy(dtype=int),
        outcome[observed],
        feature_names=args.covariates,
        reference_action=args.reference_action,
    )
    report = model.report()
    report["horizon_months"] = args.horizon
    report["excluded_immature_censoring_n"] = int((~observed).sum())
    report["iptw_km_rmst"] = iptw_rmst_contrasts(
        numeric.to_numpy(),
        pd.to_numeric(df[args.treatment_col], errors="raise").to_numpy(dtype=int),
        pd.to_numeric(df[args.time_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(df[args.event_col], errors="raise").to_numpy(dtype=int),
        reference_action=args.reference_action,
        tau=args.horizon,
        trim=args.trim,
        n_splits=args.folds,
        bootstrap_replicates=args.bootstrap,
        random_state=args.seed,
    )
    save_json(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
