# P-HELP-HCC Code Release

This directory contains a reproducible project implementation derived from
`paper/main.tex`, especially the Phase A/C/E/P method sections and the
experimental setup.

The private 673-patient HCC cohort is not included. The code supports two
data modes:

- real CSV/XLSX input with survival time, event, treatment, and clinical
  covariates;
- synthetic HCC cohort generation using the paper's published cohort
  statistics for dry runs and reproducibility checks.

## Implemented Paper Components

- dataset loading, cohort validation, survival-label derivation, and
  67-dimensional curated feature construction;
- repeated patient-level 5-fold splits across seeds
  `42, 123, 2024, 31415, 65537`;
- Phase A artificial society projection with 46-dimensional agent state;
- Phase C survival ensemble:
  regularized multinomial logistic regression, gradient boosting/XGBoost
  fallback, calibrated random forest, and PyTorch MLP with focal loss;
- K-means phenotype branch with PCA at 95% variance and selected `K=4`;
- stacking/fusion weights learned on internal validation by Brier loss;
- Cox elastic-net hazard layer implemented with PyTorch partial likelihood;
- SHAP/permutation explanation utilities and Cox-SHAP alignment report;
- counterfactual treatment sweep with propensity gate `[0.05, 0.95]`,
  guideline confidence threshold `0.6`, bootstrap support, and
  `P(OS>12m)` treatment-arm reports;
- Phase P streaming virtual-real error controller with the paper constants.

## Quick Start

From this directory:

```powershell
python -m pip install -e .
python -m p_help_hcc.data make-synthetic --out data/synthetic_hcc.csv --n 120
python -m p_help_hcc.train --config configs/default.yaml --data data/synthetic_hcc.csv --output outputs/smoke --fast
python -m p_help_hcc.test --model outputs/smoke/fold_0/model.joblib --data data/synthetic_hcc.csv --split outputs/smoke/splits_seed_42.json --fold 0
python -m p_help_hcc.validate --data data/synthetic_hcc.csv --model outputs/smoke/fold_0/model.joblib
```

The full paper-scale setting is encoded in `configs/default.yaml`. Use it
without `--fast` when the real cohort and the target workstation are available.

## Real Data Contract

Minimum required columns:

- `overall_survival_months`: observed overall survival in months;
- `event`: 1 for death/event, 0 for censored;
- optional `survival_class`: one of `0..7` or `C1..C8`; if absent it is derived;
- optional `surgical_strategy`: `none`, `ablation`, or `resection`;
- optional `dominant_aetiology`: `HBV`, `HCV`, or `NBNC`;
- optional clinical columns such as `age`, `sex_male`, `tumor_size_cm`,
  `afp`, `albumin`, `bilirubin`, `inr`, `ajcc_stage`, and treatment flags.

The preprocessing pipeline maps available clinical columns into a stable
canonical 67-feature schema. If columns named `x_00` ... `x_66` are present,
they are treated as an already-curated feature matrix. Extra free-text or
identifier columns are not serialized into model artifacts.

Do not place PHI in `data/` for release. Local real-data files under common
formats (`csv`, `tsv`, `xlsx`, `xls`, `parquet`, and raw/private folders) are
ignored by Git. Saved `joblib`/pickle model artifacts can execute code when
loaded; only load models produced locally or from a trusted release.

Excel and parquet loading require the matching pandas engine, such as
`openpyxl` for XLSX or `pyarrow` for parquet.

## Verification

Run local tests:

```powershell
python -m unittest discover -s tests
```

Run the package smoke path:

```powershell
python -m p_help_hcc.train --config configs/default.yaml --data data/synthetic_hcc.csv --output outputs/smoke --fast
```
