# IEEE IoTJ Acceptance Artifact Checklist

This checklist captures the evidence requested in the IEEE IoTJ reviewer
comments. It is not a publication guarantee, but the next review round
should not be launched until the hard-blocker items are present.

## Hard-Blocker Artifacts

| Artifact class | Expected evidence |
| --- | --- |
| Split manifests | `splits_seed_*.json` for the paper-scale seeds/folds |
| Per-fold metrics | `metrics.json` for all 25 private-cohort runs |
| Per-fold predictions | paired true/predicted probabilities or labels for each fold |
| Model artifacts | fold models/checkpoints plus model/data hashes |
| Statistical scripts | bootstrap, Nadeau--Bengio, McNemar, IPCW C-index, calibration, and decision-curve scripts |
| Figure inputs | raw tabular inputs used to generate each plotted figure |
| Censoring audit | patient-level event/censor tables, IPCW weights, and 60-month sensitivity outputs |
| Locked external validation | frozen Internal-673 model evaluated on an external cohort without retraining |
| Edge profiling | measured device logs for latency, memory, power/energy, attribution cost, scenario-sweep cost, and Phase-P logging overhead |

## Soft Artifacts

| Artifact class | Expected evidence |
| --- | --- |
| Environment lock | `environment.yml`, `requirements.txt`, `Dockerfile`, and package hashes |
| HLPL comparator | released implementation or enough run artifacts to reproduce HLPL baseline results |
| Regulatory/clinical protocol | intended-use table, IRB/silent-shadow protocol, and human-factors plan |

## Local Readiness Check

Run this before starting another review round:

```powershell
python scripts/check_acceptance_artifacts.py --root .
```

The script exits with status `1` while hard blockers are missing and status `0`
only when all hard-blocker artifact classes are detected and
`audit/manifest.json` contains the exact 25 seed/fold pairs, required schemas,
in-repository paths, and matching SHA-256 digests. Passing the script is still
only a readiness signal; reviewers must inspect the evidence quality.

## Current State

As of 2026-07-21, the local `data/` and `outputs/` directories contain only
`.gitkeep` placeholders. The manuscript and code are therefore not ready for a
third accept-gate review round.
