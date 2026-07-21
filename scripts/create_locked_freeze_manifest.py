"""Seal a fitted Internal-673 model and its inference contracts before export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_hlpl_hcc.locked_validation import build_freeze_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a locked-model freeze manifest")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-cohort", default="Internal-673")
    parser.add_argument(
        "--trusted-model",
        action="store_true",
        help="Required acknowledgement: joblib files can execute code and must be trusted.",
    )
    args = parser.parse_args(argv)
    if not args.trusted_model:
        raise ValueError("Refusing to load joblib without --trusted-model")
    if args.training_cohort != "Internal-673":
        raise ValueError("Locked external validation requires an Internal-673 model")
    model_path = Path(args.model)
    model = joblib.load(model_path)
    manifest = build_freeze_manifest(
        model, model_path, training_cohort=args.training_cohort
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
