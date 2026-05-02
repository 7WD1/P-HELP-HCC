"""Stable feature preprocessing for the paper's 67-dimensional input."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .constants import CANONICAL_COLUMNS, CURATED_DIM


EXCLUDE_COLUMNS = {
    "patient_id",
    "mrn",
    "medical_record_number",
    "name",
    "patient_name",
    "dob",
    "date_of_birth",
    "national_id",
    "phone",
    "email",
    "address",
    CANONICAL_COLUMNS["time"],
    CANONICAL_COLUMNS["event"],
    CANONICAL_COLUMNS["label"],
}

CURATED_FEATURE_NAMES = [
    "age",
    "age_ge_65",
    "sex_male",
    "year_of_diagnosis_scaled",
    "race_asian",
    "race_other",
    "diabetes",
    "comorbidity_count",
    "hbv_positive",
    "hcv_positive",
    "nbnc",
    "cirrhosis",
    "alcohol_related",
    "antiviral_therapy",
    "log_afp",
    "albumin",
    "bilirubin",
    "inr",
    "platelets",
    "meld",
    "albi_grade",
    "child_pugh_b_or_c",
    "alt",
    "ast",
    "creatinine",
    "sodium",
    "hemoglobin",
    "neutrophil_lymphocyte_ratio",
    "liver_recovery_score",
    "tumor_size_cm",
    "log_tumor_size",
    "lesion_count",
    "multifocal",
    "vascular_invasion",
    "extrahepatic_spread",
    "ajcc_stage",
    "stage_i",
    "stage_ii",
    "stage_iii",
    "stage_iv",
    "bclc_stage",
    "bclc_a",
    "bclc_b",
    "bclc_c",
    "bclc_d",
    "treatment_no_resection",
    "treatment_ablation",
    "treatment_resection",
    "treatment_tace",
    "treatment_rfa",
    "treatment_sorafenib",
    "treatment_combo",
    "radiotherapy",
    "chemotherapy",
    "palliative_care",
    "surgical_margin_positive",
    "transplant",
    "systemic_agents",
    "ecog",
    "portal_hypertension",
    "ascites",
    "encephalopathy",
    "tumor_number_gt3",
    "size_gt5",
    "afp_gt400",
    "resection_eligible",
    "aux_signal_01",
]

EXPLICIT_FEATURE_COLUMNS = [f"x_{i:02d}" for i in range(CURATED_DIM)]


def _numeric(df: pd.DataFrame, *names: str, default: float = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _binary(df: pd.DataFrame, *names: str, default: float = np.nan) -> pd.Series:
    return _numeric(df, *names, default=default).clip(0, 1)


def _contains(df: pd.DataFrame, col: str, pattern: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[col].astype(str).str.lower().str.contains(pattern, na=False).astype(float)


def _stage(df: pd.DataFrame, col: str) -> pd.Series:
    raw = _numeric(df, col, default=np.nan)
    if raw.notna().any():
        return raw
    if col in df.columns:
        mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "a": 1, "b": 2, "c": 3, "d": 4}
        return df[col].astype(str).str.lower().str.strip().map(mapping).astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _first_extra_numeric(df: pd.DataFrame, used: set[str]) -> pd.Series:
    for col in sorted(df.columns):
        if col.lower() in EXCLUDE_COLUMNS or col in used:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(0.0, index=df.index, dtype=float)


def build_curated_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in EXPLICIT_FEATURE_COLUMNS):
        return df[EXPLICIT_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    used: set[str] = set()
    age = _numeric(df, "age", "age_years")
    sex_male = _binary(df, "sex_male", "male")
    year = _numeric(df, "year_of_diagnosis", "diagnosis_year")
    hbv = _binary(df, "hbv_positive", "hbv")
    hcv = _binary(df, "hcv_positive", "hcv")
    nbnc = _binary(df, "nbnc")
    if nbnc.isna().all():
        nbnc = ((hbv.fillna(0) == 0) & (hcv.fillna(0) == 0)).astype(float)
    cirrhosis = _binary(df, "cirrhosis")
    diabetes = _binary(df, "diabetes")
    tumor_size = _numeric(df, "tumor_size_cm", "tumor_size")
    lesion_count = _numeric(df, "lesion_count", "tumor_count", default=1.0)
    ajcc = _stage(df, "ajcc_stage")
    bclc = _stage(df, "bclc_stage")
    afp = _numeric(df, "afp", "alpha_fetoprotein")
    child = _binary(df, "child_pugh_b_or_c", "child_pugh_bc")
    no_resection = _binary(df, "treatment_no_resection")
    ablation = _binary(df, "treatment_ablation")
    resection = _binary(df, "treatment_resection")
    if "surgical_strategy" in df.columns:
        strategy = df["surgical_strategy"].astype(str).str.lower()
        no_resection = no_resection.fillna(strategy.str.contains("none|no", na=False).astype(float))
        ablation = ablation.fillna(strategy.str.contains("ablation|rfa", na=False).astype(float))
        resection = resection.fillna(strategy.str.contains("resection", na=False).astype(float))

    features = {
        "age": age,
        "age_ge_65": (age >= 65).astype(float),
        "sex_male": sex_male,
        "year_of_diagnosis_scaled": (year - 2008) / 14.0,
        "race_asian": _contains(df, "race", "asian"),
        "race_other": _contains(df, "race", "other|black|white|unknown"),
        "diabetes": diabetes,
        "comorbidity_count": _numeric(df, "comorbidity_count", default=np.nan),
        "hbv_positive": hbv,
        "hcv_positive": hcv,
        "nbnc": nbnc,
        "cirrhosis": cirrhosis,
        "alcohol_related": _binary(df, "alcohol_related", "alcohol_use"),
        "antiviral_therapy": _binary(df, "antiviral_therapy"),
        "log_afp": np.log1p(afp.clip(lower=0)),
        "albumin": _numeric(df, "albumin"),
        "bilirubin": _numeric(df, "bilirubin"),
        "inr": _numeric(df, "inr"),
        "platelets": _numeric(df, "platelets", "platelet"),
        "meld": _numeric(df, "meld"),
        "albi_grade": _numeric(df, "albi_grade"),
        "child_pugh_b_or_c": child,
        "alt": _numeric(df, "alt"),
        "ast": _numeric(df, "ast"),
        "creatinine": _numeric(df, "creatinine"),
        "sodium": _numeric(df, "sodium"),
        "hemoglobin": _numeric(df, "hemoglobin"),
        "neutrophil_lymphocyte_ratio": _numeric(df, "neutrophil_lymphocyte_ratio", "nlr"),
        "liver_recovery_score": (1.0 - 0.01 * (age - 50).clip(lower=0) - 0.2 * cirrhosis.fillna(0)).clip(0, 1),
        "tumor_size_cm": tumor_size,
        "log_tumor_size": np.log1p(tumor_size.clip(lower=0)),
        "lesion_count": lesion_count,
        "multifocal": _binary(df, "multifocal"),
        "vascular_invasion": _binary(df, "vascular_invasion"),
        "extrahepatic_spread": _binary(df, "extrahepatic_spread"),
        "ajcc_stage": ajcc,
        "stage_i": (ajcc == 1).astype(float),
        "stage_ii": (ajcc == 2).astype(float),
        "stage_iii": (ajcc == 3).astype(float),
        "stage_iv": (ajcc == 4).astype(float),
        "bclc_stage": bclc,
        "bclc_a": (bclc == 1).astype(float),
        "bclc_b": (bclc == 2).astype(float),
        "bclc_c": (bclc == 3).astype(float),
        "bclc_d": (bclc == 4).astype(float),
        "treatment_no_resection": no_resection,
        "treatment_ablation": ablation,
        "treatment_resection": resection,
        "treatment_tace": _binary(df, "treatment_tace"),
        "treatment_rfa": _binary(df, "treatment_rfa"),
        "treatment_sorafenib": _binary(df, "treatment_sorafenib"),
        "treatment_combo": _binary(df, "treatment_combo"),
        "radiotherapy": _binary(df, "radiotherapy"),
        "chemotherapy": _binary(df, "chemotherapy"),
        "palliative_care": _binary(df, "palliative_care"),
        "surgical_margin_positive": _binary(df, "surgical_margin_positive"),
        "transplant": _binary(df, "transplant"),
        "systemic_agents": _binary(df, "systemic_agents"),
        "ecog": _numeric(df, "ecog", "ecog_status"),
        "portal_hypertension": _binary(df, "portal_hypertension"),
        "ascites": _binary(df, "ascites"),
        "encephalopathy": _binary(df, "encephalopathy"),
        "tumor_number_gt3": (lesion_count > 3).astype(float),
        "size_gt5": (tumor_size > 5).astype(float),
        "afp_gt400": (afp > 400).astype(float),
        "resection_eligible": ((ajcc <= 2) & (child.fillna(0) == 0) & (tumor_size <= 5)).astype(float),
        "aux_signal_01": _first_extra_numeric(df, used),
    }
    frame = pd.DataFrame({name: features[name] for name in CURATED_FEATURE_NAMES}, index=df.index)
    return frame.apply(pd.to_numeric, errors="coerce")


@dataclass
class PHelpPreprocessor:
    curated_dim: int = CURATED_DIM
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    output_names_: list[str] = field(default_factory=lambda: list(CURATED_FEATURE_NAMES))

    def fit(self, df: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "PHelpPreprocessor":
        if self.curated_dim != CURATED_DIM:
            raise ValueError(f"P-HELP-HCC expects curated_dim={CURATED_DIM}")
        curated = build_curated_feature_frame(df)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        imputed = self.imputer.fit_transform(curated)
        self.scaler.fit(imputed)
        self.output_names_ = list(curated.columns)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("PHelpPreprocessor is not fitted")
        curated = build_curated_feature_frame(df).reindex(columns=self.output_names_)
        x = self.imputer.transform(curated)
        return self.scaler.transform(x).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> np.ndarray:
        return self.fit(df, y).transform(df)
