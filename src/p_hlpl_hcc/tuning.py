"""Optional exploratory candidate-search utility.

This module is not called by the training or reproduction workflows and must
not be treated as evidence that nested cross-validation was run for the paper.
The production path uses the explicit within-fold validation stream only.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import deep_update
from .data import validate_and_prepare_dataframe
from .pipeline import PHlplHCCPipeline


def paper_search_grid() -> list[dict[str, Any]]:
    rf_estimators = [100, 300, 500]
    rf_depths = [6, 10, None]
    xgb_lrs = [0.03, 0.05, 0.1]
    xgb_estimators = [300, 500, 800]
    hidden_dims = [[128, 64], [256, 128], [256, 128, 64]]
    dropouts = [0.1, 0.2, 0.3]
    alphas = np.round(np.arange(0.0, 1.0001, 0.05), 2).tolist()
    k_values = [2, 3, 4, 5, 6]
    grid = []
    for rf_n, rf_depth, xgb_lr, xgb_n, dims, dropout, alpha, k in product(
        rf_estimators, rf_depths, xgb_lrs, xgb_estimators, hidden_dims, dropouts, alphas, k_values
    ):
        grid.append(
            {
                "phase_c": {
                    "fusion_alpha": float(alpha),
                    "random_forest": {"n_estimators": rf_n, "max_depth": rf_depth},
                    "xgboost": {"n_estimators": xgb_n, "learning_rate": xgb_lr},
                    "gradient_boosting_fallback": {"n_estimators": xgb_n, "learning_rate": xgb_lr},
                    "mlp": {"hidden_dims": dims, "dropout": dropout},
                    "clustering": {"k": k},
                }
            }
        )
    return grid


def run_inner_search(
    train_df: pd.DataFrame,
    base_config: dict[str, Any],
    candidate_overrides: Iterable[dict[str, Any]],
    *,
    seed: int = 42,
    inner_folds: int = 4,
    max_candidates: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run an opt-in exploratory inner-fold candidate comparison.

    With Phase P enabled, candidate selection uses the pipeline's explicit
    ``Macro-F1 - weight * replay residual`` score.  A6 (Phase P disabled)
    reduces exactly to validation Macro-F1.  The routine is intentionally not
    invoked by :mod:`p_hlpl_hcc.train` or the reproduction orchestrator.
    """

    df = validate_and_prepare_dataframe(
        train_df,
        time_col=base_config["data"]["target_time_col"],
        event_col=base_config["data"]["event_col"],
        label_col=base_config["data"]["label_col"],
    ).reset_index(drop=True)
    y = df[base_config["data"]["label_col"]].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    rows = []
    best_score = -float("inf")
    best_override: dict[str, Any] = {}
    for cand_idx, override in enumerate(candidate_overrides):
        if max_candidates is not None and cand_idx >= max_candidates:
            break
        config = deep_update(base_config, override)
        scores = []
        macro_scores = []
        residual_scores = []
        for fold_id, (tr, va) in enumerate(splitter.split(np.zeros(len(df)), y)):
            model = PHlplHCCPipeline(config=config, seed=seed + fold_id).fit(
                df.iloc[tr].reset_index(drop=True),
                df.iloc[va].reset_index(drop=True),
            )
            scores.append(model.selection_score_)
            macro_scores.append(model.val_metrics_["macro_f1"])
            residual_scores.append(model.phase_p_validation_residual_)
        mean_score = float(np.mean(scores))
        rows.append(
            {
                "candidate": cand_idx,
                "selection_score": mean_score,
                "macro_f1": float(np.mean(macro_scores)),
                "phase_p_residual": float(np.mean(residual_scores)),
                "override": override,
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_override = deepcopy(override)
    return best_override, pd.DataFrame(rows)

