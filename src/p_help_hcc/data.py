"""Data loading, validation, and synthetic cohort generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import CANONICAL_COLUMNS, CLASS_LABELS, SURVIVAL_CUTPOINTS_MONTHS
from .utils import ensure_dir, seed_everything


def survival_months_to_class(months: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(months), dtype=float)
    labels = np.digitize(values, SURVIVAL_CUTPOINTS_MONTHS, right=False)
    return np.clip(labels, 0, len(CLASS_LABELS) - 1).astype(int)


def normalize_label_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int)
        if values.min() >= 1 and values.max() <= 8:
            values = values - 1
        if values.min() < 0 or values.max() > 7:
            raise ValueError("survival_class must be encoded as 0..7, 1..8, or C1..C8")
        return values
    cleaned = series.astype(str).str.upper().str.strip()
    mapping = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    mapping.update({str(idx): idx for idx in range(8)})
    mapping.update({str(idx + 1): idx for idx in range(8)})
    mapped = cleaned.map(mapping)
    if mapped.isna().any():
        bad = sorted(cleaned[mapped.isna()].unique().tolist())
        raise ValueError(f"Invalid survival_class labels: {bad}")
    return mapped.astype("Int64").astype(int)


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported data file extension: {suffix}")


def validate_and_prepare_dataframe(
    df: pd.DataFrame,
    *,
    time_col: str = CANONICAL_COLUMNS["time"],
    event_col: str = CANONICAL_COLUMNS["event"],
    label_col: str = CANONICAL_COLUMNS["label"],
) -> pd.DataFrame:
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"Missing required survival time column: {time_col}")
    if event_col not in out.columns:
        raise ValueError(f"Missing required event indicator column: {event_col}")
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    event_numeric = pd.to_numeric(out[event_col], errors="coerce")
    if event_numeric.isna().any() or not set(event_numeric.dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"Column {event_col} must contain only 0/1 event indicators")
    out[event_col] = event_numeric.astype(int)
    if out[time_col].isna().any():
        raise ValueError(f"Column {time_col} contains non-numeric or missing values")
    if label_col in out.columns:
        out[label_col] = normalize_label_series(out[label_col])
    else:
        immature_censored = (out[event_col] == 0) & (out[time_col] < SURVIVAL_CUTPOINTS_MONTHS[-1])
        if immature_censored.any():
            raise ValueError(
                "Cannot derive hard eight-class survival labels for censored rows before 72 months. "
                f"Provide {label_col} or remove/handle {int(immature_censored.sum())} immature censored rows."
            )
        out[label_col] = survival_months_to_class(out[time_col].to_numpy())
    if CANONICAL_COLUMNS["surgery"] not in out.columns:
        out[CANONICAL_COLUMNS["surgery"]] = "unknown"
    if CANONICAL_COLUMNS["aetiology"] not in out.columns:
        out[CANONICAL_COLUMNS["aetiology"]] = "unknown"
    return out


def generate_synthetic_hcc_cohort(n: int = 673, seed: int = 42) -> pd.DataFrame:
    """Generate a de-identified synthetic cohort matching paper-level summaries."""

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    surgery_probs = np.array([312, 138, 223], dtype=float) / 673.0
    surgery = rng.choice(["none", "ablation", "resection"], size=n, p=surgery_probs)
    stage_by_surgery = {
        "none": [0.093, 0.199, 0.321, 0.387],
        "ablation": [0.449, 0.348, 0.145, 0.058],
        "resection": [0.354, 0.363, 0.256, 0.027],
    }
    class_by_surgery = {
        "none": [0.426, 0.228, 0.144, 0.074, 0.054, 0.035, 0.026, 0.013],
        "ablation": [0.116, 0.130, 0.152, 0.145, 0.130, 0.116, 0.145, 0.065],
        "resection": [0.076, 0.076, 0.139, 0.157, 0.135, 0.143, 0.197, 0.076],
    }
    age_params = {
        "none": (66.4, 10.5),
        "ablation": (63.0, 11.0),
        "resection": (61.2, 11.8),
    }
    tumor_params = {
        "none": (7.8, 4.2),
        "ablation": (3.4, 1.9),
        "resection": (4.1, 2.6),
    }

    rows: list[dict[str, float | int | str]] = []
    for trt in surgery:
        age = np.clip(rng.normal(*age_params[trt]), 28, 92)
        tumor_size = np.clip(rng.normal(*tumor_params[trt]), 0.5, 20.0)
        stage_idx = int(rng.choice([1, 2, 3, 4], p=stage_by_surgery[trt]))
        y = int(rng.choice(np.arange(8), p=np.array(class_by_surgery[trt]) / np.sum(class_by_surgery[trt])))
        low = [0, 6, 12, 24, 36, 48, 60, 72][y]
        high = [6, 12, 24, 36, 48, 60, 72, 96][y]
        os_months = float(rng.uniform(low + 0.1, high))
        event_prob = 0.94 - 0.05 * y
        event = int(rng.random() < np.clip(event_prob, 0.45, 0.95))
        hbv = int(rng.random() < 0.528)
        hcv = int((not hbv) and (rng.random() < 0.22))
        nbnc = int(not hbv and not hcv)
        cirrhosis = int(rng.random() < 0.684)
        male = int(rng.random() < 0.746)
        afp = float(np.exp(rng.normal(np.log(35 + 120 * stage_idx + 18 * tumor_size), 1.0)))
        albumin = float(np.clip(rng.normal(4.1 - 0.18 * stage_idx - 0.25 * cirrhosis, 0.45), 1.8, 5.3))
        bilirubin = float(np.clip(rng.lognormal(np.log(0.9 + 0.35 * stage_idx), 0.35), 0.2, 8.5))
        inr = float(np.clip(rng.normal(1.02 + 0.06 * stage_idx + 0.1 * cirrhosis, 0.12), 0.8, 2.2))
        platelets = float(np.clip(rng.normal(185 - 22 * cirrhosis - 8 * stage_idx, 45), 35, 420))
        meld = float(np.clip(6 + 1.8 * bilirubin + 4.0 * (inr - 1.0) + rng.normal(0, 1.5), 6, 32))
        row: dict[str, float | int | str] = {
            "patient_id": f"SYN{len(rows) + 1:04d}",
            "age": round(age, 3),
            "sex_male": male,
            "dominant_aetiology": "HBV" if hbv else ("HCV" if hcv else "NBNC"),
            "hbv_positive": hbv,
            "hcv_positive": hcv,
            "nbnc": nbnc,
            "cirrhosis": cirrhosis,
            "diabetes": int(rng.random() < (0.20 + 0.004 * (age - 50))),
            "tumor_size_cm": round(tumor_size, 3),
            "multifocal": int(rng.random() < {"none": 0.516, "ablation": 0.268, "resection": 0.287}[trt]),
            "lesion_count": int(max(1, rng.poisson(1.0 + 1.2 * (stage_idx >= 3)))),
            "vascular_invasion": int(rng.random() < (0.08 + 0.16 * (stage_idx - 1))),
            "extrahepatic_spread": int(stage_idx == 4 and rng.random() < 0.55),
            "ajcc_stage": stage_idx,
            "bclc_stage": stage_idx,
            "afp": round(afp, 3),
            "albumin": round(albumin, 3),
            "bilirubin": round(bilirubin, 3),
            "inr": round(inr, 3),
            "platelets": round(platelets, 3),
            "meld": round(meld, 3),
            "albi_grade": int(np.clip(round(1 + 0.32 * stage_idx + rng.normal(0, 0.35)), 1, 3)),
            "child_pugh_b_or_c": int((albumin < 3.2) or (bilirubin > 2.2) or (inr > 1.35)),
            "surgical_strategy": trt,
            "treatment_no_resection": int(trt == "none"),
            "treatment_ablation": int(trt == "ablation"),
            "treatment_resection": int(trt == "resection"),
            "treatment_tace": int(trt == "none" and rng.random() < 0.55),
            "treatment_rfa": int(trt == "ablation"),
            "treatment_sorafenib": int(stage_idx >= 3 and rng.random() < 0.35),
            "treatment_combo": int(stage_idx >= 3 and rng.random() < 0.20),
            "radiotherapy": int(rng.random() < 0.16),
            "chemotherapy": int(rng.random() < 0.22),
            "overall_survival_months": round(os_months, 3),
            "event": event,
            "survival_class": y,
        }
        # Fill additional raw-like covariates so preprocessing exercises selection.
        for j in range(1, 44):
            signal = 0.05 * stage_idx - 0.02 * y + 0.01 * (age - 60)
            row[f"aux_clinical_{j:02d}"] = round(float(rng.normal(signal, 1.0)), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P-HELP-HCC data utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("make-synthetic", help="Generate a synthetic HCC cohort")
    gen.add_argument("--out", required=True)
    gen.add_argument("--n", type=int, default=673)
    gen.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "make-synthetic":
        df = generate_synthetic_hcc_cohort(args.n, args.seed)
        out = Path(args.out)
        ensure_dir(out.parent)
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
