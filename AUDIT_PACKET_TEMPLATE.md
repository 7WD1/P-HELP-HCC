# Controlled Audit Packet Template

This template is the minimum evidence package needed before launching another
IEEE IoTJ accept-gate review. Do not include PHI in a public repository. For
private data, provide this packet through the institutionally approved
controlled-access channel.

## Directory Layout

```text
audit/
|-- README.md
|-- manifest.json
|-- environment/
|   |-- environment.yml
|   |-- requirements.txt
|   `-- docker-image-digest.txt
|-- hashes/
|   |-- data_hashes.json
|   `-- model_hashes.json
|-- splits/
|   |-- splits_seed_42.json
|   |-- splits_seed_123.json
|   |-- splits_seed_2024.json
|   |-- splits_seed_31415.json
|   `-- splits_seed_65537.json
|-- private_internal_673/
|   |-- fold_predictions/
|   |   `-- seed_<seed>_fold_<fold>_predictions.csv
|   |-- fold_metrics/
|   |   `-- seed_<seed>_fold_<fold>_metrics.json
|   |-- checkpoints/
|   |   `-- seed_<seed>_fold_<fold>_model.joblib
|   |-- censoring/
|   |   |-- event_censor_table_by_class.csv
|   |   |-- ipcw_weights_by_fold.csv
|   |   `-- sensitivity_60m_vs_72m.csv
|   `-- statistics/
|       |-- paired_macro_f1_tests.py
|       |-- mcnemar_tests.py
|       |-- ipcw_cindex_tests.py
|       |-- bootstrap_ci.py
|       `-- decision_curve_inputs.csv
|-- locked_external_validation/
|   |-- protocol.md
|   |-- frozen_preprocessing_manifest.json
|   |-- external_predictions.csv
|   |-- external_metrics.json
|   `-- calibration_by_horizon.csv
|-- public_cohort_retraining/
|   |-- tcga_lihc/
|   |-- seer/
|   `-- mimic_iv_hcc/
|-- figure_inputs/
|   |-- fig_cross_cohort_macroF1.csv
|   |-- fig_confusion_matrix.csv
|   |-- fig_calibration_brier.csv
|   |-- fig_decision_curve.csv
|   `-- ...
`-- edge_profiling/
    |-- protocol.md
    |-- jetson_orin_profile.csv
    |-- xavier_nx_profile.csv
    |-- raspberry_pi5_profile.csv
    `-- power_latency_memory_summary.json
```

## Prediction File Schema

Each `*_predictions.csv` should contain one row per test patient per fold.

| Column | Required | Description |
| --- | --- | --- |
| `patient_id` | yes | Pseudonymized stable identifier |
| `seed` | yes | Paper seed |
| `fold` | yes | Fold id |
| `true_class` | yes | `0..7` or `C1..C8` |
| `event` | yes | `1` death/event, `0` censored |
| `overall_survival_months` | yes | Observed follow-up time |
| `p_c1` ... `p_c8` | yes | Predicted class probabilities |
| `pred_class` | yes | Argmax predicted class |
| `mortality_risk_score` | yes | Score used for C-index |
| `phenotype` | recommended | Routed phenotype cluster |
| `scenario_arm` | recommended | Displayed scenario arm, if applicable |

## Metrics File Schema

Each `*_metrics.json` should include:

```json
{
  "seed": 42,
  "fold": 0,
  "macro_f1": 0.0,
  "top2_accuracy": 0.0,
  "top3_accuracy": 0.0,
  "ipcw_cindex": 0.0,
  "ece_12m": 0.0,
  "brier_12m": 0.0,
  "per_class": {
    "C1": {"sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0, "n": 0},
    "C8": {"sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0, "n": 0}
  }
}
```

## Locked External Validation Protocol

The locked external-validation folder must document:

- training data and tuning performed only on Internal-673,
- frozen preprocessing, imputation, feature mapping, thresholds, and model
  checkpoints,
- external cohort inclusion/exclusion and label mapping,
- no retraining or threshold tuning on the external cohort,
- Macro-F1, top-k accuracy, IPCW C-index, calibration, Brier score, per-class
  metrics, and failure cases.

## Edge Profiling Protocol

Measured edge profiling should include:

- device model, OS, runtime, CPU/GPU/NPU mode, and power mode,
- batch size and number of warm-up / timed runs,
- end-to-end inference latency,
- attribution latency,
- scenario-sweep latency,
- Phase-P logging overhead,
- peak memory,
- power or energy if available,
- raw logs plus summary statistics.

## Pre-Review Command

After assembling the packet under `audit/`, run:

```powershell
python scripts/check_acceptance_artifacts.py --root .
```

Only launch another six-reviewer accept-gate round after this check reports
`Ready for acceptance re-review: True`.

The manifest must conform to `schemas/audit_manifest.schema.json`, enumerate
exactly the five configured seeds times five folds, and include SHA-256 digests
for every fold metric table, prediction table, and model artifact. A filename
match without a valid schema and digest does not pass the gate.
