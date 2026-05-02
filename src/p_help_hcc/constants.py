"""Constants transcribed from the P-HELP-HCC paper."""

from __future__ import annotations

from math import inf

CLASS_LABELS = [f"C{i}" for i in range(1, 9)]
N_CLASSES = 8
CURATED_DIM = 67
AGENT_STATE_DIM = 46

SURVIVAL_CUTPOINTS_MONTHS = [6, 12, 24, 36, 48, 60, 72]
SURVIVAL_BINS_MONTHS = [0, *SURVIVAL_CUTPOINTS_MONTHS, inf]

PAPER_SEEDS = [42, 123, 2024, 31415, 65537]

ACTION_SET = ["None", "Resection", "TACE", "RFA", "Sorafenib", "Combo"]

CLASS_WEIGHTS = [1.0, 1.5, 1.7, 2.1, 2.5, 2.7, 2.3, 4.5]

SOCIETY_DIMS = {
    "patient": 12,
    "tumor": 6,
    "liver": 8,
    "treatment": 6,
    "guideline": 4,
    "explanation": 10,
}

FEATURE_BLOCK_SLICES = {
    "dem": (0, 8),
    "hep": (8, 14),
    "lab": (14, 29),
    "tum": (29, 45),
    "trt": (45, 58),
    "fu": (58, 67),
}

PHASE_A_DEFAULTS = {
    "delta_t_months": 1,
    "horizon_months": 72,
    "d_max_cm": 20.0,
    "growth_rate_lognormal_mu": -1.4,
    "growth_rate_lognormal_sigma": 0.6,
    "afp_alpha_per_month": 0.04,
    "afp_beta": 1.8,
    "fibrosis_age_rate_per_year": 0.002,
    "fibrosis_treatment_bump": 0.015,
    "fibrosis_recovery_rate": 0.020,
    "process_noise_std": 0.05,
}

PHASE_P_DEFAULTS = {
    "soft_error_threshold": 0.18,
    "hard_error_threshold": 0.32,
    "online_learning_rate": 1e-4,
    "proximal_weight": 1e-3,
    "monitor_window": 30,
    "retrain_buffer": 200,
    "classification_calibration_mix": 0.5,
}

CANONICAL_COLUMNS = {
    "time": "overall_survival_months",
    "event": "event",
    "label": "survival_class",
    "surgery": "surgical_strategy",
    "aetiology": "dominant_aetiology",
}

