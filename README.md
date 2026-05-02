<div align="center">

# P-HELP-HCC

**Parallel Hierarchical Explainable Learning Pipeline for Hepatocellular Carcinoma Survival Stratification**

Reference implementation accompanying the paper, faithful to the Phase&nbsp;A / C / E / P method sections and the eight-class HCC experimental setup.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-research--code-orange.svg)](#)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture at a Glance](#architecture-at-a-glance)
3. [Implemented Paper Components](#implemented-paper-components)
4. [Quick Start](#quick-start)
5. [Project Layout](#project-layout)
6. [Configuration & Hyperparameters](#configuration--hyperparameters)
7. [Real Data Contract](#real-data-contract)
8. [Verification](#verification)
9. [Reproducibility Notes](#reproducibility-notes)
10. [Citation](#citation)

---

## Overview

P-HELP-HCC is a parallel-system survival stratification framework for hepatocellular carcinoma (HCC). It organises the eight-class prognosis task as four ACP phases:

| Phase | Theme | Output |
|-------|-------|--------|
| **A** | Artificial society of six interacting agents | 46-dim agent state $\mathbf{S}_i$ |
| **C** | Computational experiments (survival, surgical scenarios, counterfactuals) | Class probabilities, RMST gaps, scenario effects |
| **E** | Multi-layer explainable survival model | Cox HR, SHAP, phenotype, counterfactual, feedback |
| **P** | Parallel execution and adaptive update | Streaming virtual–real error controller |

The private 673-patient cohort is **not** redistributed. The code supports two data modes:

- real CSV / TSV / XLSX / Parquet input that satisfies the [data contract](#real-data-contract);
- synthetic HCC cohort generation that reproduces the paper's published per-class and per-strategy statistics for dry runs.

---

## Architecture at a Glance

```
                                                                                    
            +-----------------+         +--------------------+   Phase E            
   x_i  --> | Phase A         | S_i --> | Phase C            | -----------+         
  67-dim    | Society         | 46-dim  | 4-learner ensemble |            v         
            | Transformer     |         | + K-means K=4      |   +-----------------+
            +-----------------+         | + counterfactual   |   | Cox-EN, SHAP,   |
                  ^                     +--------------------+   | cluster, CF,    |
                  |                              |               | feedback layers |
                  |                              v               +-----------------+
            +-----------------+         +--------------------+            |         
            | Phase P         | <------ | virtual-real error |  <---------+         
            | controller      |  retrain| streaming monitor  |   IPCW Brier        
            +-----------------+         +--------------------+                      
```

> The K-means phenotype branch operates on the **67-dim curated input** (Eq. 7), the survival ensemble and counterfactual rollouts operate on the **46-dim agent state** (Eq. 4), and the Phase P calibration loss is the **IPCW multinomial Brier** of Eq. 13 with a matured-only fallback below the numerical floor.

---

## Implemented Paper Components

- **Data layer.** Dataset loading, canonical column validation, survival-label derivation, and the curated 67-feature schema (Section 8.1).
- **Splits.** Repeated patient-level 5-fold cross-validation under five seeds `{42, 123, 2024, 31415, 65537}`, stratified jointly on the eight-class label and the surgical-strategy axis (Section 8.2).
- **Phase A.** Artificial society projection into the $d_S{=}46$ agent state, with the calibrated dynamics constants of Table II (Gompertz, AFP log-domain recursion, fibrosis update, treatment policy, Child–Pugh feasibility cap).
- **Phase C — survival ensemble.** Four heterogeneous base learners merged by Brier-optimal fusion: regularized multinomial logistic, XGBoost (Gradient Boosting fallback), calibrated random forest, PyTorch MLP with class-weighted focal loss (Section 5.1).
- **Phase C — phenotype branch.** PCA conserving 95% variance plus K-means with $K{=}4$ on the curated input, returning silhouette / Davies–Bouldin / Calinski–Harabasz internal validity indices.
- **Phase C — counterfactual sweep.** Six-action space `{None, Resection, TACE, RFA, Sorafenib, Combo}`, propensity gate $[0.05,\,0.95]$, guideline-confidence threshold $\rho^{\star}{=}0.6$, bootstrap of $B{=}200$ replicates, and $P(\mathrm{OS}{>}12\,\mathrm{m})$ treatment-arm reports (Section 5.4 + Algorithm 1).
- **Phase E.** Cox elastic-net hazard layer with PyTorch partial-likelihood, SHAP/permutation explanation utilities, Cox–SHAP rank alignment, and the explanation-consistency loss with $\tanh$-sharpness $\kappa{=}5$ (Section 6).
- **Phase P.** Streaming virtual–real error controller with thresholds $\bar{e}_{\text{soft}}{=}0.18$ / $\bar{e}_{\text{hard}}{=}0.32$, online step $\eta{=}10^{-4}$, proximal anchor $\lambda_w{=}10^{-3}$, monitor window $n_b{=}30$, retrain buffer $n_r{=}200$, and error mixing weight $\alpha_e{=}0.5$ (Section 7 + Algorithm 2).

---

## Quick Start

Install in editable mode and run a fast smoke pass against a synthetic cohort:

```powershell
python -m pip install -e .

# 1. Synthesize a small cohort that respects the paper's class / strategy mix.
python -m p_help_hcc.data make-synthetic --out data/synthetic_hcc.csv --n 120

# 2. Train the full pipeline (Phase A -> C -> E) with the --fast smoke profile.
python -m p_help_hcc.train --config configs/default.yaml `
                           --data   data/synthetic_hcc.csv `
                           --output outputs/smoke `
                           --fast

# 3. Evaluate a held-out fold.
python -m p_help_hcc.test --model outputs/smoke/fold_0/model.joblib `
                          --data  data/synthetic_hcc.csv `
                          --split outputs/smoke/splits_seed_42.json `
                          --fold  0

# 4. Run the cohort-level validation harness.
python -m p_help_hcc.validate --data  data/synthetic_hcc.csv `
                              --model outputs/smoke/fold_0/model.joblib
```

> The full paper-scale setting (5 seeds × 5 folds = 25 runs, full estimator counts, 100 MLP epochs) lives in `configs/default.yaml`. Drop the `--fast` flag once a real cohort and the target workstation are available.

---

## Project Layout

```
code/
├── configs/
│   ├── default.yaml          # Paper-scale hyperparameters
│   └── search_grid.yaml      # Nested search grid from Section 8.2
├── data/                     # PHI-free local input directory (git-ignored)
├── outputs/                  # Run artefacts (git-ignored)
├── scripts/run_smoke.ps1     # One-shot smoke runner
├── src/p_help_hcc/
│   ├── society.py            # Phase A: SocietyTransformer + dynamics
│   ├── clustering.py         # Phase C: PCA + K-means phenotype routing
│   ├── ensemble.py           # Phase C: 4-learner Brier-optimal stacking
│   ├── neural.py             # Phase C: MLP with focal loss
│   ├── counterfactual.py     # Phase C: scenario sweep + propensity gate
│   ├── cox.py                # Phase E: Cox elastic-net (Torch)
│   ├── explain.py            # Phase E: SHAP + IPCW Brier + L_exp / L_clin
│   ├── parallel.py           # Phase P: streaming error controller
│   ├── pipeline.py           # End-to-end PHelpHCCPipeline
│   ├── splits.py             # Repeated 5-fold patient splits
│   ├── preprocessing.py      # 67-feature curation + imputation
│   ├── data.py               # Loading + synthetic cohort generator
│   └── train.py / test.py / validate.py
└── tests/                    # Unit + smoke tests
```

---

## Configuration & Hyperparameters

The released `configs/default.yaml` encodes the paper's final selections. The most consulted values are:

| Block | Symbol | Value |
|-------|:------:|------:|
| Curated input dim | $d$ | $67$ |
| Agent state dim | $d_S$ | $46$ |
| Outer / inner CV | folds × seeds | $5 \times 5$ |
| Phenotypes | $K_c^{\star}$ | $4$ |
| MLP backbone | hidden / dropout / activation | $[256,128,64]$ / $0.2$ / GELU |
| Optimizer | Adam, lr, batch, epochs, patience | $10^{-3}$, $32$, $100$, $15$ |
| Focal loss | $\gamma$ | $1.5$ |
| Random Forest | $n_{\text{est}}$, max depth | $500$, $10$ |
| XGBoost | $n_{\text{est}}$, lr, max depth | $500$, $0.05$, $6$ |
| Fusion | $\alpha_{\text{fuse}}$ | $0.60$ |
| Counterfactual | $B$, propensity gate, $\rho^{\star}$ | $200$, $[0.05,0.95]$, $0.6$ |
| Phase E loss | $\gamma_1{=}\lambda_{\text{cal}}$ / $\gamma_2{=}\lambda_{\text{exp}}$ / $\gamma_3{=}\lambda_{\text{clin}}$ | $1.0$ / $0.2$ / $0.1$ |
| Phase E loss | $\tanh$ sharpness $\kappa$ | $5.0$ |
| Cox elastic-net | epochs, lr, $\lambda_{\ell_1}$, $\lambda_{\ell_2}$ | $300$, $0.03$, $10^{-3}$, $10^{-3}$ |
| Phase P thresholds | $\bar e_{\text{soft}}$ / $\bar e_{\text{hard}}$ | $0.18$ / $0.32$ |
| Phase P windows | $n_b$ / $n_r$ | $30$ / $200$ |
| Class weights | C1 … C8 | `[1.0, 1.5, 1.7, 2.1, 2.5, 2.7, 2.3, 4.5]` |

The nested grid for hyperparameter search lives in `configs/search_grid.yaml` and matches Section 8.2 cell-for-cell.

---

## Real Data Contract

**Required columns**

| Column | Type | Description |
|---|---|---|
| `overall_survival_months` | float | Observed overall survival in months from diagnosis |
| `event` | int (0/1) | $1$ for death/event, $0$ for censored |

**Optional columns (auto-derived if absent)**

| Column | Allowed values |
|---|---|
| `survival_class` | `0..7` or `C1..C8` |
| `surgical_strategy` | `none` / `ablation` / `resection` |
| `dominant_aetiology` | `HBV` / `HCV` / `NBNC` |
| Clinical covariates | `age`, `sex_male`, `tumor_size_cm`, `afp`, `albumin`, `bilirubin`, `inr`, `ajcc_stage`, treatment flags, … |

The preprocessing pipeline maps available clinical columns into a stable canonical 67-feature schema (`x_00` … `x_66`). When those `x_*` columns are already present they are treated as a curated feature matrix. Free-text or identifier columns are never serialized into the model artefacts.

> **Data hygiene.** Do **not** place PHI under `data/` for release. Common tabular formats and `data/raw/` & `data/private/` are ignored by Git out-of-the-box. Saved `joblib`/pickle model artefacts can execute arbitrary code when loaded; only load models produced locally or from a trusted release.

Excel and Parquet loading require the matching pandas engine, e.g. `openpyxl` for `.xlsx` or `pyarrow` for `.parquet`.

---

## Verification

Local tests:

```powershell
python -m unittest discover -s tests
```

Smoke pipeline (synthetic cohort, `--fast` budget):

```powershell
python -m p_help_hcc.train --config configs/default.yaml `
                           --data   data/synthetic_hcc.csv `
                           --output outputs/smoke `
                           --fast
```

Each fold writes `metrics.json`, the trained `model.joblib`, and a per-fold split manifest. Run-level `metrics.csv` and `metrics_summary.json` aggregate the 25 paper runs.

---

## Reproducibility Notes

- Bit-level reproducibility uses a master seed of `42` propagated to NumPy, PyTorch (CPU/CUDA), Python's `random`, scikit-learn, and XGBoost; deterministic CUDA is enabled via `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- The 25-run protocol (5 seeds × 5 folds) is reported as mean ± std and was paired with $1{,}000$-resample percentile bootstrap CIs that agreed with the std-based intervals to within $\pm 0.005$.
- The Phase P controller is a **retrospective emulation** of the parallel execution loop. Prospective deployment would require a future silent shadow run before the soft / hard threshold rules take effect.

---

## Citation

If you use this code or build upon the framework, please cite the paper:

```bibtex
@article{phelp_hcc,
  title   = {P-HELP-HCC: Parallel Hierarchical Explainable Learning Pipeline
             for Hepatocellular Carcinoma Survival Stratification},
  author  = {Anonymous},
  journal = {Manuscript under review},
  year    = {2026}
}
```

---

<div align="center">

For methodological background see the accompanying manuscript (`paper/main.tex`).<br/>
Issues and reproducibility questions are welcome via the GitHub tracker.

</div>
