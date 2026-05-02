"""Train and cross-validate P-HELP-HCC."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import apply_fast_overrides, load_config
from .data import generate_synthetic_hcc_cohort, load_table, validate_and_prepare_dataframe
from .pipeline import PHelpHCCPipeline
from .splits import build_repeated_splits
from .utils import ensure_dir, save_json, seed_everything


def _fold_dir(output: Path, seeds: list[int], seed: int, fold_id: int) -> Path:
    if len(seeds) == 1:
        return output / f"fold_{fold_id}"
    return output / f"seed_{seed}" / f"fold_{fold_id}"


def run_training(config: dict, data_path: str | None, output: str, *, fast: bool = False) -> pd.DataFrame:
    if fast:
        config = apply_fast_overrides(config)
    output_dir = ensure_dir(output)
    if data_path:
        df = load_table(data_path)
    else:
        df = generate_synthetic_hcc_cohort(n=673, seed=42)
        generated_path = output_dir / "synthetic_hcc.csv"
        df.to_csv(generated_path, index=False)
    df = validate_and_prepare_dataframe(
        df,
        time_col=config["data"]["target_time_col"],
        event_col=config["data"]["event_col"],
        label_col=config["data"]["label_col"],
    )
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
            model = PHelpHCCPipeline(config=config, seed=seed).fit(train_df, val_df)
            test_metrics = model.evaluate(test_df)
            joblib.dump(model, fdir / "model.joblib")
            metrics = {
                "seed": seed,
                "fold": fold_id,
                **{f"train_{k}": v for k, v in model.train_metrics_.items()},
                **{f"val_{k}": v for k, v in model.val_metrics_.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"cluster_{k}": v for k, v in model.phenotype_quality(train_df).items()},
                **{f"phase_e_{k}": v for k, v in model.phase_e_loss_.items()},
            }
            save_json(metrics, fdir / "metrics.json")
            rows.append(metrics)
            print(f"seed={seed} fold={fold_id} test_macro_f1={test_metrics['macro_f1']:.4f}")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    summary = metrics_df.drop(columns=["seed", "fold"]).agg(["mean", "std"]).to_dict()
    save_json(summary, output_dir / "metrics_summary.json")
    return metrics_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train P-HELP-HCC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="CSV/XLSX/parquet data. If omitted, a synthetic cohort is generated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fast", action="store_true", help="Use reduced folds/epochs/estimators for smoke testing.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_training(config, args.data, args.output, fast=args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
