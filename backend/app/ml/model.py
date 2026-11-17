"""
Backend XGBoost inference. Loads the joblib model plus its meta file and refuses to run if the meta's
feature schema does not match the code's (catches train/serve skew at startup).

Probabilities are post-processed with the temperature fitted at training time (meta.calibration);
the browser does the identical maths (frontend/src/lib/edge-math.ts) and CI checks they agree.
Explanations are exact TreeSHAP values from XGBoost (pred_contribs), in log-odds units of the predicted class.
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from backend.app.ml.feature_schema import (
    CLASS_TO_ESI,
    FEATURE_LABELS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    build_feature_vector,
)

logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "triage_model.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "triage_model.meta.json")
CARD_PATH = os.path.join(ARTIFACT_DIR, "model_card.json")

TOP_DRIVERS = 4


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T). T > 1 softens over-confident models; T == 1 is the identity. argmax never changes."""
    if temperature == 1.0:
        return probs
    q = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temperature)
    return q / q.sum(axis=-1, keepdims=True)


@dataclass
class Contribution:
    feature: str
    label: str
    value: Optional[float]
    contribution: float  # TreeSHAP, log-odds of the predicted class; positive = pushes toward that ESI


@dataclass
class MLResult:
    esi: int
    probabilities: List[float]  # calibrated
    top_probability: float  # calibrated
    margin: float
    model_version: str
    label_source: str
    contributions: List[Contribution] = field(default_factory=list)


class TriageModel:
    def __init__(self, model: Any, meta: dict):
        if [f["name"] for f in meta["features"]] != FEATURE_NAMES:
            raise ValueError("Model meta feature order does not match ml/feature_schema.py. Retrain the model.")
        if meta["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Model feature_schema_version does not match the code. Retrain the model.")
        if meta["class_to_esi"] != CLASS_TO_ESI:
            raise ValueError("Model class_to_esi mapping does not match the code.")
        self.model = model
        self.meta = meta
        self.version: str = meta["model_version"]
        self.label_source: str = meta.get("label_source", "unknown")
        self.temperature: float = float(meta.get("calibration", {}).get("temperature", 1.0))

    def _matrix(self, values: Mapping[str, Any]) -> np.ndarray:
        return np.array([build_feature_vector(values)], dtype=np.float32)

    def predict(self, values: Mapping[str, Any], explain: bool = True) -> MLResult:
        x = self._matrix(values)
        probs = apply_temperature(self.model.predict_proba(x), self.temperature)[0]
        order = np.argsort(probs)[::-1]
        top, second = float(probs[order[0]]), float(probs[order[1]]) if len(order) > 1 else 0.0
        result = MLResult(
            esi=CLASS_TO_ESI[int(order[0])], probabilities=[float(p) for p in probs], top_probability=top,
            margin=top - second, model_version=self.version, label_source=self.label_source,
        )
        if explain:
            try:
                result.contributions = self.explain(values, x, int(order[0]))
            except Exception as e:  # explanations must never break an assessment
                logger.warning("Explanation failed: %s", e)
        return result

    def explain(self, values: Mapping[str, Any], x: Optional[np.ndarray] = None, class_idx: Optional[int] = None) -> List[Contribution]:
        import xgboost as xgb

        x = self._matrix(values) if x is None else x
        contribs = self.model.get_booster().predict(xgb.DMatrix(x, missing=np.nan), pred_contribs=True)  # (1, classes, feats+1)
        k = int(np.argmax(self.model.predict_proba(x)[0])) if class_idx is None else class_idx
        row = contribs[0][k][:-1]  # last column is the bias term
        out: List[Contribution] = []
        for i in np.argsort(-np.abs(row))[:TOP_DRIVERS]:
            name = FEATURE_NAMES[int(i)]
            raw = values.get(name)
            out.append(Contribution(
                feature=name, label=FEATURE_LABELS[name],
                value=None if raw is None or (isinstance(raw, float) and math.isnan(raw)) else float(raw),
                contribution=round(float(row[int(i)]), 4),
            ))
        return [c for c in out if abs(c.contribution) > 1e-6]


@lru_cache(maxsize=1)
def get_model() -> Optional[TriageModel]:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        logger.warning("Triage model artifacts missing in %s; ML stage disabled.", ARTIFACT_DIR)
        return None
    import joblib

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return TriageModel(joblib.load(MODEL_PATH), meta)


def get_model_card() -> Optional[Dict[str, Any]]:
    if not os.path.exists(CARD_PATH):
        return None
    with open(CARD_PATH, encoding="utf-8") as f:
        return json.load(f)
