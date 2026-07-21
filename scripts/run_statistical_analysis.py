"""Run paired audit statistics from real out-of-fold prediction/metric tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.statistics import (
    mcnemar_exact,
    nadeau_bengio_corrected_ttest,
    paired_ipcw_c_index_test,
)
from p_hlpl_hcc.utils import save_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired repeated-CV and prediction tests")
    parser.add_argument("--fold-metrics", required=True)
    parser.add_argument("--paired-predictions", required=True)
    parser.add_argument("--model-metric-col", required=True)
    parser.add_argument("--comparator-metric-col", required=True)
    parser.add_argument("--truth-col", default="true_class")
    parser.add_argument("--model-pred-col", required=True)
    parser.add_argument("--comparator-pred-col", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--time-col")
    parser.add_argument("--event-col")
    parser.add_argument("--model-risk-col")
    parser.add_argument("--comparator-risk-col")
    parser.add_argument("--cindex-bootstrap", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    fold = pd.read_csv(args.fold_metrics)
    paired = pd.read_csv(args.paired_predictions)
    result = {
        "nadeau_bengio": nadeau_bengio_corrected_ttest(
            fold[args.model_metric_col].to_numpy(float)
            - fold[args.comparator_metric_col].to_numpy(float),
            test_fraction=args.test_fraction,
        ),
        "mcnemar": mcnemar_exact(
            paired[args.truth_col].to_numpy(),
            paired[args.model_pred_col].to_numpy(),
            paired[args.comparator_pred_col].to_numpy(),
        ),
        "inputs": {
            "fold_rows": int(len(fold)),
            "paired_patient_rows": int(len(paired)),
        },
    }
    cindex_columns = [
        args.time_col,
        args.event_col,
        args.model_risk_col,
        args.comparator_risk_col,
    ]
    if any(cindex_columns) and not all(cindex_columns):
        parser.error(
            "time/event/model-risk/comparator-risk columns must be supplied together"
        )
    if all(cindex_columns):
        result["ipcw_c_index"] = paired_ipcw_c_index_test(
            paired[args.time_col].to_numpy(float),
            paired[args.event_col].to_numpy(int),
            paired[args.model_risk_col].to_numpy(float),
            paired[args.comparator_risk_col].to_numpy(float),
            replicates=args.cindex_bootstrap,
        )
    save_json(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
