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
    "tumor_growth_prior_mean": 0.07,
    "tumor_growth_prior_std": 0.03,
    "afp_alpha_per_month": 0.04,
    "afp_beta": 1.8,
    "fibrosis_kappa_age_per_month": 0.005,
    "fibrosis_kappa_treatment_per_month": 0.010,
    "fibrosis_kappa_recovery_per_month": 0.015,
    "fibrosis_age_rate_per_year": 0.002,
    "fibrosis_treatment_bump": 0.015,
    "fibrosis_recovery_rate": 0.020,
    "process_noise_std": 0.05,
}

PHASE_P_DEFAULTS = {
    "soft_error_threshold": 0.18,
    "hard_error_threshold": 0.32,
    "abstention_entropy_soft": 0.65,
    "abstention_entropy_hard": 0.85,
    "online_learning_rate": 5e-3,
    "proximal_weight": 1e-2,
    "monitor_window": 30,
    "retrain_buffer": 200,
    "classification_calibration_mix": 0.5,
}

PHASE_E_LOSS_DEFAULTS = {
    "lambda_cal": 0.4,
    "lambda_exp": 0.3,
    "lambda_clin": 0.2,
    "tanh_kappa": 5.0,
}

GUIDELINE_CONFIDENCE_CUTOFF = 0.30

PCA_VARIANCE_RETAINED = 0.90

DP_SGD_DEFAULTS = {
    "epsilon_per_round": 4.0,
    "delta": 1e-5,
    "noise_multiplier_sigma": 1.1,
    "l2_clip_C": 1.0,
}

MAHALANOBIS_OOD_PERCENTILE = 99

CANONICAL_COLUMNS = {
    "time": "overall_survival_months",
    "event": "event",
    "label": "survival_class",
    "surgery": "surgical_strategy",
    "aetiology": "dominant_aetiology",
}

