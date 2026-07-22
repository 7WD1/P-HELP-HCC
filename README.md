# P-HLPL-HCC

**Parallel Explainable Internet of Medical Things Framework with a Structured Multi-Agent Patient-State Representation for Hepatocellular Carcinoma Survival Prediction**

Reference implementation accompanying the manuscript. The release contains executable analysis paths and non-evidentiary smoke tests; it does not redistribute the private 673-patient cohort or claim that absent paper-run artifacts have been reproduced.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-research--code-orange.svg)](#)

**English** | [Traditional Chinese](README.zh-TW.md)

**Authors.** Wen-Dong Jiang (Ningbo University, `wendongjiang@ieee.org`); Tsung-Jung Lin (Tamkang University and Taipei City Hospital, Ren-Ai Branch, `dab70@tpech.gov.tw`); Chih-Yung Chang (Tamkang University, `cychang@mail.tku.edu.tw`); Diptendu Sinha Roy (National Institute of Technology, Shillong, `diptendu.sr@nitm.ac.in`).

**Manuscript submitted to the IEEE Internet of Things Journal (IEEE IoTJ).**

## Contents

1. [Overview](#overview)
2. [Architecture at a Glance](#architecture-at-a-glance)
3. [Implemented Paper Components](#implemented-paper-components)
4. [Quick Start](#quick-start)
5. [Project Layout](#project-layout)
6. [Configuration and Hyperparameters](#configuration-and-hyperparameters)
7. [Real Data Contract](#real-data-contract)
8. [Verification](#verification)
9. [Reproducibility Notes](#reproducibility-notes)
10. [Citation](#citation)

## Overview

P-HLPL-HCC is a parallel-system survival stratification framework for hepatocellular carcinoma (HCC). It organizes the eight-class prognosis task as four ACP phases:

| Phase | Theme | Output |
|-------|-------|--------|
| **A** | Structured representation with six deterministic, separately maskable state blocks | 46-dim patient state `S_i` |
| **C** | Computational experiments for survival and bounded treatment scenarios | Class probabilities, RMST gaps, observational scenario effects |
| **E** | Multi-layer explainable survival model | Cox HR, attribution proxies, phenotype, scenario consistency |
| **P** | Parallel execution replay monitor | Silent-shadow residual logs and review triggers |

The private 673-patient cohort is **not** redistributed. The code separates two data uses:

- authorized CSV / TSV / XLSX / Parquet input that satisfies the [data contract](#real-data-contract) for analyses;
- fixture-only records for schema, unit, and smoke software tests. Fixture records are not clinical observations and must never be used for paper experiments, metric reproduction, or scientific inference.

## Architecture at a Glance

```text
   x_i  --> +-----------------+ S_i --> +--------------------+ --> Phase E
   67-dim   | Phase A         | 46-dim  | Phase C            |     +-----------------+
            | deterministic   |         | 4-learner ensemble |     | Cox-EN, attrib.,|
            | projector       |         | K-means K=4        |     | cluster, scen., |
            +-----------------+         | scenario sweep     |     | replay layers   |
                    ^                   +--------------------+     +-----------------+
                    |                            |                         |
                    |                            v                         |
            +-----------------+         +--------------------+            |
            | Phase P         | <------ | replay residual    | <----------+
            | monitor         | review  | silent-shadow log  | IPCW Brier
            +-----------------+         +--------------------+
```

The K-means phenotype branch operates on the **67-dim curated input**. The survival ensemble and scenario sweeps operate on the **46-dim patient state**. Phase P uses `1-P(true)+alpha_e*multiclass-Brier` as its replay residual and combines it with censoring-informed/IPCW sample weighting; hard-classification rows require a mature endpoint.

## Implemented Paper Components

- **Data layer.** Dataset loading, canonical column validation, survival-label derivation, and the curated 67-feature schema.
- **Splits.** Repeated patient-level 5-fold cross-validation under five seeds `{42, 123, 2024, 31415, 65537}`, stratified jointly on the eight-class label and event indicator (with deterministic fallbacks when a joint stratum is too small).
- **Phase A.** Static structured projection into the `d_S=46` six-block patient state. Equations expressed in centimeters and months are never applied to this standardized state. Gompertz/fibrosis updates are confined to the separate physical-unit longitudinal back-test and require source-scale inputs; no separate AFP transition is implemented in the released back-test.
- **Phase C - survival ensemble.** Four heterogeneous base learners merged by Brier-optimal fusion: regularized multinomial logistic, XGBoost (Gradient Boosting fallback), calibrated random forest, and PyTorch MLP with class-weighted focal loss.
- **Phase C - phenotype branch.** PCA conserving 90% variance plus K-means with `K=4` on the curated input, returning silhouette, Davies-Bouldin, and Calinski-Harabasz internal validity indices. The resulting one-hot cluster indicator enters fusion calibration only; it never routes or gates separate experts.
- **Phase C - scenario head and observational analyses.** The MLP uses a shared-backbone six-action auxiliary head during training (`scenario_auxiliary.loss_weight=0.20`). The separate single-patient survival sweep uses pretreatment Patient/Tumor/Liver state for its propensity gate, neutralizes factual-treatment-derived auxiliary coordinates for every arm, consumes exactly `B=200` externally generated finite patient-bootstrap prediction draws plus guideline confidences, applies `rho*=0.30`, and otherwise returns an empty clinical display. The cohort-level CLI reports naive, IPTW, and cross-fitted AIPW/DR contrasts; overlap retention; standardized mean differences; E-values; and IPTW-Kaplan--Meier RMST contrasts. Cohort summaries use `B=1000` patient resamples; the RMST bootstrap refits the propensity model on every draw, while the AIPW score bootstrap explicitly keeps cross-fitted nuisance predictions fixed. All outputs are observational sensitivity summaries, not causal effects or recommendations.
- **Phase E.** A state-aligned Cox elastic-net direction is fitted before the neural ensemble. `L_cal` (one Phase-P sample weight per patient times multiclass Brier), `L_exp` (differentiable state-risk input-gradient versus normalized Cox-direction alignment), and `L_clin` (selected-state gradient-sign hinge) are added only to the MLP objective and back-propagated with configurable weights `0.4/0.3/0.2`; named switches set each contribution to zero. Tree and logistic branches retain their native objectives. Post-hoc permutation attribution and externally supplied attribution-alignment diagnostics remain separate; the repository does not implement SHAP-in-the-loop.
- **Phase P.** Three executable paths are connected: validation-replay residuals modulate censoring-informed training weights, the same residual enters MLP validation-stream checkpoint selection, and a one-vs-rest Platt calibrator is called by `predict_proba`. No nested hyperparameter search is invoked by the released training or reproduction commands. The replay residual is `1-P(true)+alpha_e*sum_c(P_c-onehot_c)^2`. The retrospective monitor retains thresholds `e_soft=0.18` and `e_hard=0.32`, abstention entropies `p_soft=0.65` and `p_hard=0.85`, online step `eta_w=5e-3`, proximal anchor `lambda_w=1e-2`, monitor window `n_b=30`, retrain buffer `n_r=200`, and `alpha_e=0.5`. These thresholds raise review flags only in retrospective replay; they are not validated live-control rules.
- **Deployment safeguards.** `safeguards.py` exposes research-code stubs for safeguards discussed in the paper: future federated DP-SGD, Mahalanobis-distance OOD detection, hash-chained audit logging, and a silent-shadow gate that requires explicit IRB / SaMD review before live clinical influence.

## Quick Start

Install in editable mode and run a fast software-only smoke pass against a fixture file:

```powershell
python -m pip install -e .

# 1. Create records used only to exercise schemas and software paths.
python -m p_hlpl_hcc.data make-fixture --out data/fixture_hcc.csv --n 120

# 2. Train the pipeline with the --fast smoke profile.
python -m p_hlpl_hcc.train --config configs/default.yaml `
                           --data   data/fixture_hcc.csv `
                           --output outputs/smoke `
                           --fast

# Full/A1--A6 and the three one-at-a-time Phase-P mechanism paths are named
# and executable. Smoke outputs are not evidence.
python -m p_hlpl_hcc.train --config configs/default.yaml --ablation A1 `
                           --data data/fixture_hcc.csv --output outputs/a1 --fast
python -m p_hlpl_hcc.train --config configs/default.yaml --ablation A5 `
                           --data data/fixture_hcc.csv --output outputs/a5 --fast
python -m p_hlpl_hcc.train --config configs/default.yaml --ablation A6 `
                           --data data/fixture_hcc.csv --output outputs/a6 --fast
python -m p_hlpl_hcc.train --config configs/default.yaml --ablation PhasePNoIPCW `
                           --data data/fixture_hcc.csv --output outputs/p_no_ipcw --fast
python -m p_hlpl_hcc.train --config configs/default.yaml --ablation PhasePNoCheckpoint `
                           --data data/fixture_hcc.csv --output outputs/p_no_checkpoint --fast
python -m p_hlpl_hcc.train --config configs/default.yaml --ablation PhasePNoPlatt `
                           --data data/fixture_hcc.csv --output outputs/p_no_platt --fast

# Censoring-aware discrete-time sensitivity path.
python -m p_hlpl_hcc.train --config configs/discrete_time.yaml `
                           --data data/fixture_hcc.csv --output outputs/discrete --fast

# 3. Evaluate a held-out fold.
python -m p_hlpl_hcc.test --model outputs/smoke/fold_0/model.joblib `
                          --data  data/fixture_hcc.csv `
                          --split outputs/smoke/splits_seed_42.json `
                          --fold  0

# 4. Run the cohort-level validation harness.
python -m p_hlpl_hcc.validate --data  data/fixture_hcc.csv `
                              --model outputs/smoke/fold_0/model.joblib
```

The full paper-scale setting (5 seeds x 5 folds = 25 runs, full estimator counts, 100 MLP epochs) lives in `configs/default.yaml`. Drop the `--fast` flag only when a real cohort and the target workstation are available.

## Project Layout

```text
P-HLPL-HCC/
|-- configs/
|   |-- ablations.yaml     # Full/A1--A6 plus independent Phase-P switches
|   |-- dynamics.yaml      # Four physical-unit longitudinal back-test switches
|   |-- discrete_time.yaml # Censoring-aware likelihood path
|   |-- default.yaml       # Paper-scale hyperparameters
|   `-- search_grid.yaml   # Optional candidate grid; not invoked by paper runs
|-- data/                  # PHI-free local input directory (git-ignored)
|-- outputs/               # Run artifacts (git-ignored)
|-- scripts/
|   `-- run_smoke.ps1      # One-shot smoke runner
|-- src/p_hlpl_hcc/
|   |-- society.py         # Phase A: deterministic six-block state projection
|   |-- clustering.py      # Phase C: PCA + K-means calibration feature
|   |-- ensemble.py        # Phase C: 4-learner Brier-optimal stacking
|   |-- neural.py          # Phase C: MLP with focal loss
|   |-- counterfactual.py  # Suppressed single-patient observational sweep
|   |-- observational.py   # Pretreatment-only cross-fit AIPW + patient bootstrap
|   |-- survival.py        # Discrete-time risk masks and likelihood
|   |-- discrete_pipeline.py # Executable censoring-aware survival head
|   |-- cox.py             # Phase E: Cox elastic-net (Torch)
|   |-- explain.py         # Phase E: post-hoc attribution/IPCW diagnostics
|   |-- parallel.py        # Phase P: replay residual monitor + abstention
|   |-- safeguards.py      # DP-SGD / Mahalanobis OOD / audit log / IRB gate
|   |-- pipeline.py        # End-to-end PHlplHCCPipeline
|   |-- splits.py          # Repeated 5-fold patient splits
|   |-- preprocessing.py   # 67-feature curation + imputation
|   |-- data.py            # Loading + fixture-only record builder
|   |-- train.py
|   |-- test.py
|   `-- validate.py
`-- tests/                 # Unit + smoke tests
```

## Configuration and Hyperparameters

The released `configs/default.yaml` encodes the current paper selections. The most consulted values are:

| Block | Symbol | Value |
|-------|:------:|------:|
| Curated input dim | `d` | `67` |
| Patient state dim | `d_S` | `46` |
| Repeated outer CV | folds x seeds | `5 x 5` |
| Within-fold validation stream | train/validation fraction | `0.8 / 0.2` |
| Phenotypes | `K_c*` | `4` |
| MLP backbone | hidden / dropout / activation | `[256,128,64]` / `0.2` / GELU |
| Optimizer | Adam, lr, batch, epochs, patience | `1e-3`, `32`, `100`, `15` |
| Focal loss | `gamma` | `1.5` |
| Random Forest | `n_est`, max depth | `500`, `10` |
| XGBoost | `n_est`, lr, max depth | `500`, `0.05`, `6` |
| Fusion | `alpha_fuse` | `0.60` |
| PCA variance retained | `r_PCA` | `0.90` |
| Observational analysis | patient bootstrap `B`, cross-fit folds, `rho_trim` | `1000`, `5`, `0.05` |
| Single-patient scenario sensitivity | patient bootstrap target `B`, guideline `rho*` | `200`, `0.30` |
| Phase E loss | `gamma1` / `gamma2` / `gamma3` | `0.4` / `0.3` / `0.2` |
| Phase E loss | `tanh` sharpness `kappa` | `5.0` |
| Cox elastic-net | epochs, lr, `lambda_l1`, `lambda_l2` | `300`, `0.03`, `1e-3`, `1e-3` |
| Phase P residual triggers | `e_soft` / `e_hard` | `0.18` (review recalibration) / `0.32` (review retraining) |
| Phase P abstention entropies | `p_soft` / `p_hard` | `0.65` / `0.85` |
| Phase P online step / drift penalty | `eta_w` / `lambda_w` | `5e-3` / `1e-2` |
| Phase P windows | `n_b` / `n_r` | `30` / `200` |
| Class weights | C1 to C8 | `[1.0, 1.5, 1.7, 2.1, 2.5, 2.7, 2.3, 4.5]` |
| DP-SGD (future design only) | per-round epsilon / planned total / delta / sigma / clip | `0.4` / `4.0` / `1e-5` / `1.1` / `1` |
| Mahalanobis OOD | per-class centroid percentile | `99` |

`configs/search_grid.yaml` is an optional exploratory candidate list. It is not invoked by `train`, `reproduce`, or the reported validation-stream checkpoint path and is not evidence of a completed nested-CV analysis.

## Real Data Contract

**Required columns**

| Column | Type | Description |
|---|---|---|
| `overall_survival_months` | float | Observed follow-up in months from the index HCC decision encounter |
| `event` | int (0/1) | `1` for death/event, `0` for censored |

**Optional columns (auto-derived if absent)**

| Column | Allowed values |
|---|---|
| `survival_class` | `0..7` or `C1..C8` |
| `surgical_strategy` | `none` / `ablation` / `resection` |
| `dominant_aetiology` | `HBV` / `HCV` / `NBNC` |
| Clinical covariates | `age`, `sex_male`, `tumor_size_cm`, `afp`, `albumin`, `bilirubin`, `inr`, `ajcc_stage`, index-encounter treatment-plan flags, `planned_margin_risk`, `baseline_auxiliary_risk_score`, and related structured baseline fields |

The prediction landmark is the index HCC decision encounter. Every predictor, including treatment-plan fields, must be recorded at or before that landmark; post-landmark follow-up, outcome-derived fields, and postoperative pathology such as `surgical_margin_positive` are excluded. `planned_margin_risk` denotes a preoperative assessment, not an observed surgical result. The preprocessing pipeline maps named clinical columns into a stable canonical 67-feature schema (`x_00` to `x_66`) without selecting arbitrary numeric fallback columns. When those `x_*` columns are already present they are treated as a curated feature matrix, so the caller must enforce the same landmark contract before invocation. Free-text or identifier columns are never serialized into the model artifacts.

**Data hygiene.** Do not place PHI under `data/` for release. Common tabular formats and `data/raw/` / `data/private/` are ignored by Git out of the box. Saved `joblib` or pickle model artifacts can execute arbitrary code when loaded; only load models produced locally or from a trusted release.

Excel and Parquet loading require the matching pandas engine, such as `openpyxl` for `.xlsx` or `pyarrow` for `.parquet`.

## Verification

Local tests:

```powershell
python -m unittest discover -s tests
```

Smoke pipeline (fixture-only records, `--fast` budget):

```powershell
python -m p_hlpl_hcc.train --config configs/default.yaml `
                           --data   data/fixture_hcc.csv `
                           --output outputs/smoke `
                           --fast
```

Each fold writes `metrics.json`, `predictions.csv`, `mechanism_trace.json`, `model.joblib`, and a SHA-256 model manifest. A discrete-time run also writes all seven interval-hazard columns; C8 is the remaining survival tail after the 72-month boundary. The run root records the resolved configuration, split manifests, aggregate metrics, and a data hash when a source file is supplied. `mechanism_trace.json` records which Phase-E and Phase-P training/inference paths were actually active.

The deterministic one-command wrapper is:

```powershell
python scripts/reproduce.py --data <authorized-cohort.csv> `
  --output-root outputs/reproduction --include-ablations --include-discrete-time
```

The command ends by running the conservative acceptance gate. It intentionally exits nonzero while the controlled packet is incomplete.

Additional real-input-only paths are:

```powershell
python scripts/run_observational_analysis.py --data <cohort.csv> `
  --covariates age tumor_size_cm albumin --treatment-col treatment_index `
  --reference-action 0 --output outputs/observational.json

python scripts/run_dynamics_backtest.py --data <longitudinal.csv> `
  --variant full_dynamics --output-dir outputs/dynamics

python scripts/run_locked_external_validation.py --model <frozen.joblib> `
  --freeze-manifest <freeze.json> --data <external.csv> --cohort <name> `
  --output-dir outputs/locked_external --trusted-model
```

Acceptance-gate artifact readiness:

```powershell
python scripts/check_acceptance_artifacts.py --root .
```

This check is conservative and is expected to fail until the controlled audit
packet, locked external validation outputs, censoring sensitivity files, and
measured edge-device profiling logs are present. See
`ACCEPTANCE_ARTIFACTS.md` for the reviewer-facing checklist and
`AUDIT_PACKET_TEMPLATE.md` for the expected controlled-access packet layout.

## Reproducibility Notes

- Bit-level reproducibility uses a master seed of `42` propagated to NumPy, PyTorch, Python's `random`, scikit-learn, and XGBoost. Deterministic CUDA is enabled via `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- The code configures the 25-run protocol, A1--A6, PatientOnly, six agent-drop variants, and executable cumulative/single-term loss variants; an IPCW discrete-time likelihood with seven hazards and a 72-month tail; four physical-unit-only longitudinal back-test variants; pretreatment-only observational analyses; locked-model evaluation; paired statistical tests including a patient-bootstrap IPCW C-index contrast; hashes; schemas; and a one-command orchestrator. It does not claim a completed nested-CV study. The repository still lacks the real private split manifests, paper-run predictions/checkpoints, longitudinal inputs, external-cohort inputs, figure-source tables, and measured device logs required to verify the reported numbers.
- The three Phase-P mechanism variants are `PhasePNoIPCW`, `PhasePNoCheckpoint`, and `PhasePNoPlatt`. The one-command workflow includes them under `--include-ablations`; each disables exactly one prediction-coupled path while leaving the other two enabled.
- Patient-level scenario contrasts require the recorded pretreatment action as the factual arm. The public API rejects a missing or unrecognized action instead of inferring it from the nearest treatment-state template.
- FHIR-R4 resource serialization, terminology binding, MQTT transport, and a live edge message bus are not implemented or configured in this repository. They remain deployment-design specifications and must not be described as released executable mappings.
- The Phase P monitor is evaluated through retrospective replay. Prospective deployment would require a future silent-shadow run before the soft / hard threshold rules take effect.

## Citation

If you use this code or build upon the framework, please cite the paper:

```bibtex
@article{phelp_hcc,
  title   = {Parallel Explainable Internet of Medical Things Framework with
             a Structured Multi-Agent Patient-State Representation for
             Hepatocellular Carcinoma Survival Prediction},
  author  = {Wen-Dong Jiang and Tsung-Jung Lin and Chih-Yung Chang and Diptendu Sinha Roy},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Submitted}
}
```
