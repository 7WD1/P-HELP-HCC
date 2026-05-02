"""Repeated patient-level split generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, train_test_split

from .constants import CANONICAL_COLUMNS, PAPER_SEEDS
from .data import load_table, validate_and_prepare_dataframe
from .utils import ensure_dir, save_json


def _safe_stratification(df: pd.DataFrame, cols: list[str], n_splits: int) -> np.ndarray:
    candidates = [cols, cols[:2], cols[:1]]
    for candidate in candidates:
        existing = [c for c in candidate if c in df.columns]
        if not existing:
            continue
        key = df[existing].astype(str).agg("|".join, axis=1)
        counts = key.value_counts()
        if counts.min() >= n_splits:
            return key.to_numpy()
    label = df[CANONICAL_COLUMNS["label"]].astype(str)
    if label.value_counts().min() >= n_splits:
        return label.to_numpy()
    return np.zeros(len(df), dtype=int)


def build_repeated_splits(
    df: pd.DataFrame,
    *,
    seeds: list[int] | None = None,
    outer_folds: int = 5,
    val_fraction: float = 0.2,
    stratify_cols: list[str] | None = None,
) -> dict[str, dict[str, list[dict[str, list[int]]]]]:
    seeds = seeds or PAPER_SEEDS
    stratify_cols = stratify_cols or [
        CANONICAL_COLUMNS["label"],
        CANONICAL_COLUMNS["surgery"],
        CANONICAL_COLUMNS["aetiology"],
    ]
    out: dict[str, dict[str, list[dict[str, list[int]]]]] = {}
    groups = df["patient_id"].astype(str).to_numpy() if "patient_id" in df.columns else np.arange(len(df)).astype(str)
    has_repeated_groups = len(np.unique(groups)) < len(groups)
    for seed in seeds:
        y_strat = _safe_stratification(df, stratify_cols, outer_folds)
        if has_repeated_groups:
            try:
                splitter = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(np.zeros(len(df)), y_strat, groups)
            except ValueError:
                splitter = GroupKFold(n_splits=outer_folds)
                split_iter = splitter.split(np.zeros(len(df)), y_strat, groups)
        elif len(np.unique(y_strat)) > 1:
            splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.zeros(len(df)), y_strat)
        else:
            splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.zeros(len(df)))
        folds: list[dict[str, list[int]]] = []
        for fold_id, (train_val_idx, test_idx) in enumerate(split_iter):
            train_val = df.iloc[train_val_idx]
            inner_strat = _safe_stratification(train_val, stratify_cols, 2)
            train_idx_rel, val_idx_rel = _inner_train_val_split(
                train_val,
                train_val_idx,
                inner_strat,
                seed=seed + fold_id,
                val_fraction=val_fraction,
            )
            folds.append(
                {
                    "fold": fold_id,
                    "train": train_val_idx[train_idx_rel].astype(int).tolist(),
                    "val": train_val_idx[val_idx_rel].astype(int).tolist(),
                    "test": test_idx.astype(int).tolist(),
                }
            )
        out[str(seed)] = {"folds": folds}
    return out


def _inner_train_val_split(
    train_val: pd.DataFrame,
    train_val_idx: np.ndarray,
    inner_strat: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rel_idx = np.arange(len(train_val_idx))
    groups = train_val["patient_id"].astype(str).to_numpy() if "patient_id" in train_val.columns else rel_idx.astype(str)
    has_repeated_groups = len(np.unique(groups)) < len(groups)
    if has_repeated_groups:
        n_splits = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
        n_splits = min(n_splits, len(np.unique(groups)))
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_rel, val_rel = next(splitter.split(rel_idx, inner_strat, groups))
            return train_rel, val_rel
        except ValueError:
            splitter = GroupKFold(n_splits=n_splits)
            train_rel, val_rel = next(splitter.split(rel_idx, inner_strat, groups))
            return train_rel, val_rel
    strat_arg = inner_strat if len(np.unique(inner_strat)) > 1 else None
    try:
        return train_test_split(rel_idx, test_size=val_fraction, random_state=seed, stratify=strat_arg)
    except ValueError:
        return train_test_split(rel_idx, test_size=val_fraction, random_state=seed, stratify=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create P-HELP-HCC split manifests")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=PAPER_SEEDS)
    args = parser.parse_args(argv)
    df = validate_and_prepare_dataframe(load_table(args.data))
    split_manifest = build_repeated_splits(df, seeds=args.seeds, outer_folds=args.folds)
    out_dir = ensure_dir(args.out_dir)
    for seed, manifest in split_manifest.items():
        save_json(manifest, Path(out_dir) / f"splits_seed_{seed}.json")
    print(f"wrote split manifests to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
