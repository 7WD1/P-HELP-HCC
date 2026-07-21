"""Train and cross-validate P-HLPL-HCC."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .config import apply_fast_overrides, apply_named_ablation, load_config
from .baseline import CoxPHSurvivalPipeline
from .data import load_table, validate_and_prepare_dataframe
from .discrete_pipeline import DiscreteTimeSurvivalPipeline
from .pipeline import PHlplHCCPipeline
from .splits import build_repeated_splits
from .utils import ensure_dir, save_json, seed_everything


def _fold_dir(output: Path, seeds: list[int], seed: int, fold_id: int) -> Path:
    if len(seeds) == 1:
        return output / f"fold_{fold_id}"
    return output / f"seed_{seed}" / f"fold_{fold_id}"


def run_training(config: dict, data_path: str, output: str, *, fast: bool = False) -> pd.DataFrame:
    if fast:
        config = apply_fast_overrides(config)
    output_dir = ensure_dir(output)
    if not data_path:
        raise ValueError(
            "An explicit path to an authorized cohort is required. Training never creates "
            "records automatically; software tests must create and pass a fixture file."
        )
    df = load_table(data_path)
    objective = str(config.get("data", {}).get("objective", "hard_classification")).lower()
    censoring_aware = objective in {"discrete_time", "censoring_aware_discrete_time"}
    pipeline_kind = str(config.get("experiment", {}).get("pipeline", "p_hlpl_hcc")).lower()
    df = validate_and_prepare_dataframe(
        df,
        time_col=config["data"]["target_time_col"],
        event_col=config["data"]["event_col"],
        label_col=config["data"]["label_col"],
        require_unambiguous_hard_labels=not (censoring_aware or pipeline_kind == "coxph_baseline"),
    )
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    save_json({"algorithm": "sha256", "data_file": str(Path(data_path).name), "digest": digest}, output_dir / "data_hash.json")
    splits_cfg = config.get("splits", {})
    seeds = list(map(int, splits_cfg.get("seeds", [42])))
    manifests = build_repeated_splits(
        df,
        seeds=seeds,
        outer_folds=int(splits_cfg.get("outer_folds", 5)),
        val_fraction=float(splits_cfg.get("inner_val_fraction", 0.2)),
        stratify_cols=list(splits_cfg.get("stratify_on", [])),
    )
    for seed, manifest in manifests.items():
        save_json(manifest, output_dir / f"splits_seed_{seed}.json")
    rows = []
    for seed in seeds:
        seed_everything(seed)
        for fold in manifests[str(seed)]["folds"]:
            fold_id = int(fold["fold"])
            fdir = ensure_dir(_fold_dir(output_dir, seeds, seed, fold_id))
            train_df = df.iloc[fold["train"]].reset_index(drop=True)
            val_df = df.iloc[fold["val"]].reset_index(drop=True)
            test_df = df.iloc[fold["test"]].reset_index(drop=True)
            if pipeline_kind == "coxph_baseline":
                model = CoxPHSurvivalPipeline(config=config, seed=seed).fit(train_df, val_df)
            elif censoring_aware:
                model = DiscreteTimeSurvivalPipeline(config=config, seed=seed).fit(train_df, val_df)
            else:
                model = PHlplHCCPipeline(config=config, seed=seed).fit(train_df, val_df)
            test_metrics = model.evaluate(test_df)
            model_path = fdir / "model.joblib"
            joblib.dump(model, model_path)
            model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            save_json({"algorithm": "sha256", "model_file": model_path.name, "digest": model_digest}, fdir / "model_hash.json")
            proba = model.predict_proba(test_df)
            label_col = config["data"]["label_col"]
            prediction_table = pd.DataFrame(
                {
                    "patient_id": test_df["patient_id"].astype(str).to_numpy()
                    if "patient_id" in test_df.columns
                    else [f"row_{idx}" for idx in fold["test"]],
                    "seed": seed,
                    "fold": fold_id,
                    "true_class": test_df[label_col].astype("Int64"),
                    "event": test_df[config["data"]["event_col"]].to_numpy(dtype=int),
                    "overall_survival_months": test_df[
                        config["data"]["target_time_col"]
                    ].to_numpy(dtype=float),
                    "pred_class": np.argmax(proba, axis=1),
                }
            )
            for class_index in range(proba.shape[1]):
                prediction_table[f"p_c{class_index + 1}"] = proba[:, class_index]
            if hasattr(model, "predict_hazards"):
                hazards = model.predict_hazards(test_df)
                cuts = list(map(float, getattr(model, "cutpoints_", [])))
                starts = [0.0, *cuts[:-1]]
                for interval_index, (start, stop) in enumerate(zip(starts, cuts)):
                    prediction_table[
                        f"hazard_{start:g}_{stop:g}m"
                    ] = hazards[:, interval_index]
            prediction_table.to_csv(fdir / "predictions.csv", index=False)
            base_metrics = {
                "seed": seed,
                "fold": fold_id,
            }
            if pipeline_kind == "coxph_baseline":
                metrics = {
                    **base_metrics,
                    "objective": "coxph_breslow_baseline",
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            elif censoring_aware:
                metrics = {
                    **base_metrics,
                    "objective": "censoring_aware_discrete_time",
                    "best_validation_nll": model.best_validation_nll_,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            else:
                metrics = {
                    **base_metrics,
                    "objective": "hard_classification_complete_case",
                    "selection_score": model.selection_score_,
                    "phase_p_validation_residual": model.phase_p_validation_residual_,
                    **{f"train_{k}": v for k, v in model.train_metrics_.items()},
                    **{f"val_{k}": v for k, v in model.val_metrics_.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **{f"cluster_{k}": v for k, v in model.phenotype_quality(train_df).items()},
                    **{f"phase_e_{k}": v for k, v in model.phase_e_loss_.items()},
                }
            save_json(model.mechanism_trace_, fdir / "mechanism_trace.json")
            save_json(metrics, fdir / "metrics.json")
            rows.append(
                {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }
            )
            macro_f1 = test_metrics.get("macro_f1")
            suffix = f" test_macro_f1={macro_f1:.4f}" if macro_f1 is not None else ""
            print(f"seed={seed} fold={fold_id}{suffix}")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    summary = metrics_df.drop(columns=["seed", "fold"], errors="ignore").agg(["mean", "std"]).to_dict()
    save_json(summary, output_dir / "metrics_summary.json")
    return metrics_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train P-HLPL-HCC")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data",
        required=True,
        help="Authorized cohort table in CSV/XLSX/parquet format (always required).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fast", action="store_true", help="Use reduced folds/epochs/estimators for smoke testing.")
    parser.add_argument(
        "--ablation",
        default=None,
        help="Named executable variant from configs/ablations.yaml (full or A1-A6).",
    )
    parser.add_argument(
        "--ablations-config",
        default=None,
        help="Optional path to the named-ablation manifest.",
    )
    parser.add_argument(
        "--dynamics",
        default=None,
        help=(
            "Rejected for classifier training because the main state is standardized. "
            "Use scripts/run_dynamics_backtest.py with physical-unit longitudinal data."
        ),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.ablation:
        manifest = Path(args.ablations_config) if args.ablations_config else Path(args.config).with_name("ablations.yaml")
        config = apply_named_ablation(config, manifest, args.ablation)
    if args.dynamics:
        parser.error(
            "--dynamics cannot be applied to classifier training: centimeter/month rules "
            "must not receive standardized latent states. Use "
            "scripts/run_dynamics_backtest.py with physical-unit longitudinal inputs."
        )
    run_training(config, args.data, args.output, fast=args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
