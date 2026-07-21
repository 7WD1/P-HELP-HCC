"""Evaluate a saved P-HLPL-HCC fold model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import load_table, validate_and_prepare_dataframe
from .utils import load_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test a saved P-HLPL-HCC model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args(argv)
    model = joblib.load(args.model)
    df = validate_and_prepare_dataframe(
        load_table(args.data),
        time_col=model.config["data"]["target_time_col"],
        event_col=model.config["data"]["event_col"],
        label_col=model.config["data"]["label_col"],
    )
    if args.split:
        manifest = load_json(args.split)
        fold = manifest["folds"][args.fold]
        df = df.iloc[fold["test"]].reset_index(drop=True)
    metrics = model.evaluate(df)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

