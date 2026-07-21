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
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    parent = data.pop("extends", None)
    if parent is not None:
        parent_path = (config_path.parent / str(parent)).resolve()
        if parent_path == config_path.resolve():
            raise ValueError(f"Config cannot extend itself: {config_path}")
        data = deep_update(load_config(parent_path), data)
    return data


def apply_named_variant(
    config: dict[str, Any],
    manifest_path: str | Path,
    name: str,
    *,
    experiment_key: str,
) -> dict[str, Any]:
    """Apply an executable named override to a base config."""

    manifest = load_config(manifest_path)
    variants = manifest.get("variants", manifest)
    if not isinstance(variants, dict):
        raise ValueError("Ablation manifest must contain a 'variants' mapping")
    lookup = {str(key).lower(): str(key) for key in variants}
    canonical = lookup.get(str(name).lower())
    if canonical is None:
        choices = ", ".join(sorted(map(str, variants)))
        raise ValueError(f"Unknown ablation '{name}'. Available variants: {choices}")
    override = variants[canonical]
    if not isinstance(override, dict):
        raise ValueError(f"Ablation '{canonical}' must contain a configuration mapping")
    resolved = deep_update(config, override)
    resolved.setdefault("experiment", {})[experiment_key] = canonical
    return resolved


def apply_named_ablation(
    config: dict[str, Any], manifest_path: str | Path, name: str
) -> dict[str, Any]:
    """Apply an executable named component ablation override."""

    return apply_named_variant(
        config, manifest_path, name, experiment_key="ablation"
    )


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
            "observational_analysis": {"patient_bootstrap_replicates": 25},
        },
        "phase_e": {"cox": {"epochs": 40}},
    }
    return deep_update(config, override)
