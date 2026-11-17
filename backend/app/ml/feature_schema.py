"""
ONE feature schema for training, backend inference and the browser (ONNX) model.

- Training (backend/training) builds columns in FEATURE_NAMES order.
- Backend inference (ml/model.py) builds vectors with build_feature_vector().
- The exporter writes this schema into triage_model.meta.json; the frontend (edge-math.ts) reads it, so
  feature order, derived features and the class->ESI mapping are never hard-coded twice.

Missing values are NaN (XGBoost and the ONNX tree ensemble route NaN down a learned branch), AND each
vital also has an explicit `<vital>_missing` indicator so the model (and the logistic baseline) can use
"not measured" as information instead of silently treating it as a normal value.

v3 (Phase 2): adds the six *_missing indicators.
"""
import math
from typing import Any, Dict, List, Mapping

FEATURE_SCHEMA_VERSION = "3"

_BASE: List[Dict[str, Any]] = [
    {"name": "age_years", "unit": "years", "min": 0.0, "max": 125.0, "label": "Age"},
    {"name": "heart_rate", "unit": "bpm", "min": 20, "max": 300, "label": "Heart rate"},
    {"name": "systolic_bp", "unit": "mmHg", "min": 40, "max": 300, "label": "Systolic BP"},
    {"name": "diastolic_bp", "unit": "mmHg", "min": 20, "max": 200, "label": "Diastolic BP"},
    {"name": "respiratory_rate", "unit": "breaths/min", "min": 4, "max": 80, "label": "Respiratory rate"},
    {"name": "oxygen_saturation", "unit": "%", "min": 40.0, "max": 100.0, "label": "SpO2"},
    {"name": "temperature_celsius", "unit": "C", "min": 30.0, "max": 45.0, "label": "Temperature"},
    {"name": "pain_score", "unit": "0-10", "min": 0, "max": 10, "label": "Pain score"},
    # Server: 1 when indexed history documents exist for the MRN (computed, never a checkbox).
    # MIMIC training data: 1 when the patient had an earlier ED stay. Same meaning ("we know this patient"),
    # different source; see docs/ML.md.
    {"name": "has_prior_history", "unit": "bool", "min": 0, "max": 1, "label": "Prior history on file"},
]

MISSINGNESS_SOURCES = [
    "heart_rate", "systolic_bp", "respiratory_rate", "oxygen_saturation", "temperature_celsius", "pain_score",
]
_LABELS = {f["name"]: f["label"] for f in _BASE}

_DERIVED: List[Dict[str, Any]] = [
    {"name": f"{src}_missing", "unit": "bool", "min": 0, "max": 1, "derived_from": src, "kind": "is_missing",
     "label": f"{_LABELS[src]} not measured"}
    for src in MISSINGNESS_SOURCES
]

FEATURES: List[Dict[str, Any]] = _BASE + _DERIVED
FEATURE_NAMES: List[str] = [f["name"] for f in FEATURES]
RAW_FEATURE_NAMES: List[str] = [f["name"] for f in _BASE]
FEATURE_LABELS: Dict[str, str] = {f["name"]: f["label"] for f in FEATURES}

# Model output class index -> ESI level. Index 0 is ESI 1. Decoding MUST go through this list.
CLASS_TO_ESI: List[int] = [1, 2, 3, 4, 5]


def esi_to_class(esi: int) -> int:
    return CLASS_TO_ESI.index(esi)


def class_to_esi(idx: int) -> int:
    return CLASS_TO_ESI[idx]


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def build_feature_vector(values: Mapping[str, Any]) -> List[float]:
    """Order features per the schema; None/NaN -> NaN; derived indicators computed from their source."""
    out: List[float] = []
    for f in FEATURES:
        if f.get("kind") == "is_missing":
            out.append(1.0 if _is_null(values.get(f["derived_from"])) else 0.0)
            continue
        v = values.get(f["name"])
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        out.append(math.nan if _is_null(v) else float(v))
    return out


def schema_as_dict() -> Dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": FEATURES,
        "class_to_esi": CLASS_TO_ESI,
        "missing_value": "NaN",
    }
