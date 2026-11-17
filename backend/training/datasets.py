"""
Dataset adapters. Every adapter returns the same frame so the rest of the pipeline never knows the source:

  columns: RAW_FEATURE_NAMES (+ NaN for missing) , esi (1..5), group (patient id: keeps one patient's
           visits on one side of every split), complaint (free text, may be ""), sex ("M"/"F"/NaN)

label_source says where `esi` came from and is carried into the model card and the promotion gate:
  synthetic-rules  labels produced by if/else rules on the features themselves. Plumbing tests ONLY: a model
                   trained on them mostly re-learns the rules, and any metric on them is circular.
  mimic-iv-ed      nurse-assigned ESI at triage (credentialed PhysioNet access). The real target.
  ktas-korea       Korean Triage and Acuity Scale (expert-assigned). Similar 5-level scale, NOT identical to ESI.
"""
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from backend.app.ml.feature_schema import FEATURES, RAW_FEATURE_NAMES

LABEL_SOURCES = ("synthetic-rules", "mimic-iv-ed", "ktas-korea")
_RANGES = {f["name"]: (f["min"], f["max"]) for f in FEATURES if "derived_from" not in f}
VITAL_COLUMNS = [c for c in RAW_FEATURE_NAMES if c not in ("age_years", "has_prior_history")]


@dataclass
class Dataset:
    df: pd.DataFrame
    name: str
    label_source: str
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        """Content hash of features + labels + groups (not free text), recorded with every model."""
        cols = RAW_FEATURE_NAMES + ["esi", "group"]
        h = pd.util.hash_pandas_object(self.df[cols].reset_index(drop=True), index=False).values.tobytes()
        return hashlib.sha256(h).hexdigest()


def _finish(df: pd.DataFrame, name: str, label_source: str, info: Dict[str, Any]) -> Dataset:
    """Plausibility filter (out-of-range -> NaN, never a dropped row), label validation, column order."""
    oor: Dict[str, int] = {}
    for col, (lo, hi) in _RANGES.items():
        if col == "has_prior_history":
            continue
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        oor[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    n0 = len(df)
    df = df[df["esi"].isin([1, 2, 3, 4, 5]) & df["age_years"].notna()].copy()
    df["esi"] = df["esi"].astype(int)
    info = {**info, "n_rows": len(df), "n_dropped_no_label_or_age": n0 - len(df), "n_out_of_range_set_missing": oor,
            "missing_rate": {c: round(float(df[c].isna().mean()), 4) for c in VITAL_COLUMNS},
            "esi_distribution": {int(k): int(v) for k, v in df["esi"].value_counts().sort_index().items()}}
    for col in ("complaint", "sex"):
        if col not in df:
            df[col] = "" if col == "complaint" else np.nan
    df["complaint"] = df["complaint"].fillna("").astype(str)
    return Dataset(df[RAW_FEATURE_NAMES + ["esi", "group", "complaint", "sex"]].reset_index(drop=True), name, label_source, info)


# ------------------------------------------------------------------ synthetic (plumbing only)


def synthetic_label(a, h, s, rr, sp, t, p, hx) -> int:
    if sp < 82 or s <= 65 or h < 35:
        return 1
    if sp <= 92 or s < 88 or h > 135 or (a >= 12 and (rr > 30 or rr < 8)) or (a <= 0.25 and t >= 38.0) or (a >= 65 and s < 95 and h > 110):
        return 2
    if h > 105 or s > 165 or t >= 38.5 or (p >= 7 and (hx == 1 or a > 50)):
        return 3
    if p >= 4 or t > 37.8 or hx == 1:
        return 4
    return 5


def load_synthetic(n: int = 8000, seed: int = 42, missing_rate: float = 0.15) -> Dataset:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age_years": rng.uniform(0.1, 90.0, n),
        "heart_rate": rng.normal(82, 22, n).clip(30, 220),
        "systolic_bp": rng.normal(122, 24, n).clip(50, 240),
        "diastolic_bp": rng.normal(78, 14, n).clip(30, 140),
        "respiratory_rate": rng.normal(17, 5, n).clip(6, 50),
        "oxygen_saturation": rng.normal(97, 4, n).clip(65, 100),
        "temperature_celsius": rng.normal(37.0, 0.9, n).clip(34.0, 41.5),
        "pain_score": rng.choice(range(0, 11), size=n, p=[0.2] + [0.08] * 10).astype(float),
        "has_prior_history": rng.choice([0.0, 1.0], size=n, p=[0.45, 0.55]),
    })
    df["esi"] = [
        synthetic_label(r.age_years, r.heart_rate, r.systolic_bp, r.respiratory_rate, r.oxygen_saturation,
                        r.temperature_celsius, r.pain_score, r.has_prior_history)
        for r in df.itertuples()
    ]  # labels use the TRUE values, before masking
    for col in VITAL_COLUMNS:
        df.loc[rng.random(n) < missing_rate, col] = np.nan
    df["group"] = np.arange(n)
    return _finish(df, f"synthetic-n{n}-seed{seed}", "synthetic-rules", {"note": "rule-generated labels; plumbing only"})


# ------------------------------------------------------------------ MIMIC-IV-ED


def _parse_pain(x: Any) -> float:
    try:
        v = float(str(x).strip())
    except ValueError:
        return math.nan  # 'unable', 'UA', 'critical', ...
    return v if 0 <= v <= 10 else math.nan  # MIMIC contains data-entry values like 13


def load_mimic_ed(root: str) -> Dataset:
    """root contains ed/triage.csv(.gz), ed/edstays.csv(.gz) and hosp/patients.csv(.gz) (MIMIC-IV-ED + MIMIC-IV).
    Works on the full credentialed release and on the open-access demo (tiny; for smoke tests only)."""

    def read(rel: str) -> pd.DataFrame:
        for suffix in (".csv.gz", ".csv"):
            p = os.path.join(root, rel + suffix)
            if os.path.exists(p):
                return pd.read_csv(p)
        raise FileNotFoundError(f"{rel}.csv(.gz) not found under {root}")

    triage, stays, patients = read("ed/triage"), read("ed/edstays"), read("hosp/patients")
    stays["intime"] = pd.to_datetime(stays["intime"])
    stays = stays.sort_values(["subject_id", "intime"])
    # "has prior history": the patient had an earlier ED stay in this release (first stays are censored).
    stays["has_prior_history"] = (stays.groupby("subject_id").cumcount() > 0).astype(float)

    df = triage.merge(stays[["stay_id", "intime", "gender", "has_prior_history"]], on="stay_id", how="inner")
    df = df.merge(patients[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left")
    df["age_years"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
    df = df.rename(columns={
        "heartrate": "heart_rate", "resprate": "respiratory_rate", "o2sat": "oxygen_saturation",
        "sbp": "systolic_bp", "dbp": "diastolic_bp", "acuity": "esi", "chiefcomplaint": "complaint", "gender": "sex",
    })
    df["temperature_celsius"] = (pd.to_numeric(df["temperature"], errors="coerce") - 32.0) * 5.0 / 9.0  # MIMIC stores deg F
    df["pain_score"] = df["pain"].map(_parse_pain)
    for col in ("heart_rate", "respiratory_rate", "oxygen_saturation", "systolic_bp", "diastolic_bp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["esi"] = pd.to_numeric(df["esi"], errors="coerce")
    df["group"] = df["subject_id"]
    return _finish(df, os.path.basename(os.path.normpath(root)), "mimic-iv-ed", {"source": root})


# ------------------------------------------------------------------ KTAS (Korean, public)

_KTAS_ALIASES = {
    "age_years": ["age"], "sex": ["sex"], "heart_rate": ["hr"], "systolic_bp": ["sbp"], "diastolic_bp": ["dbp"],
    "respiratory_rate": ["rr"], "temperature_celsius": ["bt"], "oxygen_saturation": ["saturation", "spo2"],
    "pain_score": ["nrs_pain"], "esi": ["ktas_expert"], "complaint": ["chief_complain", "chief_complaint"],
}


def load_ktas(path: str) -> Dataset:
    """Korean ED triage CSV (Kaggle 'Emergency Service Triage Application'). Columns are matched
    case-insensitively through aliases; if the file differs from what we expect you get a clear error."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    lower = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    missing = []
    for target, aliases in _KTAS_ALIASES.items():
        src = next((lower[a] for a in aliases if a in lower), None)
        if src is None:
            missing.append(target)
        else:
            out[target] = df[src]
    if missing:
        raise ValueError(f"KTAS file lacks columns for {missing}. Found: {list(df.columns)}")
    for col in RAW_FEATURE_NAMES:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")  # '??' and similar -> NaN
    out["esi"] = pd.to_numeric(out["esi"], errors="coerce")
    out["sex"] = pd.to_numeric(out["sex"], errors="coerce").map({1: "F", 2: "M"})  # 1=female, 2=male in this file
    out["has_prior_history"] = np.nan  # not recorded in this dataset
    out["group"] = np.arange(len(out))  # no patient id: each row is its own group
    return _finish(out, os.path.basename(path), "ktas-korea", {"source": path, "caveat": "KTAS is not ESI; use as external check only"})


def load(name: str, path: Optional[str] = None, **kw) -> Dataset:
    if name == "synthetic":
        return load_synthetic(**kw)
    if name == "mimic":
        if not path:
            raise ValueError("--path is required for the mimic dataset (folder containing ed/ and hosp/).")
        return load_mimic_ed(path)
    if name == "ktas":
        if not path:
            raise ValueError("--path is required for the ktas dataset (CSV file).")
        return load_ktas(path)
    raise ValueError(f"Unknown dataset {name!r}")
