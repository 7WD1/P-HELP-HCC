"""Configuration loading helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""

    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def apply_fast_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a small, deterministic config for smoke tests."""

    override = {
        "splits": {"outer_folds": 2, "seeds": [42]},
        "phase_c": {
            "random_forest": {"n_estimators": 20, "max_depth": 6},
            "xgboost": {"n_estimators": 20, "max_depth": 3},
            "gradient_boosting_fallback": {"n_estimators": 20, "max_depth": 2},
            "mlp": {"hidden_dims": [32, 16], "epochs": 3, "patience": 2, "batch_size": 16},
            "clustering": {"n_init": 3},
            "counterfactual": {"bootstrap_replicates": 10},
        },
        "phase_e": {"cox": {"epochs": 40}, "shap": {"background_samples": 20}},
    }
    return deep_update(config, override)

