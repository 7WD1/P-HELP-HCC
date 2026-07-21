"""Validate data schema and saved model artifacts."""

from __future__ import annotations

import argparse
import json

import joblib

from .data import load_table, validate_and_prepare_dataframe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate P-HLPL-HCC data/model readiness")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    df = validate_and_prepare_dataframe(load_table(args.data))
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "class_counts": {str(k): int(v) for k, v in df["survival_class"].value_counts().sort_index().items()},
        "event_rate": float(df["event"].mean()),
        "time_min": float(df["overall_survival_months"].min()),
        "time_max": float(df["overall_survival_months"].max()),
    }
    if args.model:
        model = joblib.load(args.model)
        sample = df.head(min(5, len(df))).copy()
        proba = model.predict_proba(sample)
        report["model_loaded"] = True
        report["sample_probability_shape"] = list(proba.shape)
        report["sample_predicted_classes"] = [int(x) for x in proba.argmax(axis=1)]
        report["counterfactual_first_patient"] = model.counterfactual_report(sample, row=0)[:3]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

