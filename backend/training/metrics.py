"""
Metrics that matter for triage. Accuracy alone hides the failure that hurts patients: UNDER-triage.

ESI numbers: lower = more acute. So "under-triage" means predicted ESI > true ESI.
  high_acuity_sensitivity  of patients whose label is ESI 1-2, the share predicted ESI 1-2 (the key safety metric)
  under_triage / severe    predicted less acute than the label / by 2+ levels
  over_triage              predicted more acute than the label (costs resources, not lives)
Probability metrics (NLL, Brier, ECE) are computed against the LABEL SOURCE, not against clinical truth.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

N_CLASSES = 5
MIN_SUBGROUP_N = 30


def expected_calibration_error(proba: np.ndarray, y_class: np.ndarray, bins: int = 10) -> float:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_class[m]).mean() - conf[m].mean())
    return float(ece)


def reliability_bins(proba: np.ndarray, y_class: np.ndarray, bins: int = 10) -> List[Dict]:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_confidence": round(float(conf[m].mean()), 4),
                        "accuracy": round(float((pred[m] == y_class[m]).mean()), 4)})
    return out


def evaluate(y_esi: np.ndarray, pred_esi: np.ndarray, proba: Optional[np.ndarray] = None) -> Dict:
    y, p = np.asarray(y_esi), np.asarray(pred_esi)
    n = len(y)
    out: Dict = {"n": int(n)}
    if n == 0:
        return out
    high = y <= 2
    diff = p - y
    out.update({
        "accuracy": round(float((p == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0)), 4),
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic")), 4) if len(set(y)) > 1 else None,
        "under_triage": round(float((diff > 0).mean()), 4),
        "severe_under_triage": round(float((diff >= 2).mean()), 4),
        "over_triage": round(float((diff < 0).mean()), 4),
        "high_acuity_n": int(high.sum()),
        "high_acuity_sensitivity": round(float((p[high] <= 2).mean()), 4) if high.any() else None,
        "recall_by_esi": {str(k): (round(float((p[y == k] == k).mean()), 4) if (y == k).any() else None) for k in range(1, 6)},
    })
    if proba is not None:
        yc = y - 1
        eps = 1e-12
        onehot = np.eye(N_CLASSES)[yc]
        out.update({
            "nll": round(float(-np.log(np.clip(proba[np.arange(n), yc], eps, 1)).mean()), 4),
            "brier": round(float(((proba - onehot) ** 2).sum(axis=1).mean()), 4),
            "ece": round(expected_calibration_error(proba, yc), 4),
        })
    return out


def subgroup_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    age = df["age_years"].to_numpy()
    critical = df[["heart_rate", "systolic_bp", "respiratory_rate", "oxygen_saturation"]].isna().any(axis=1).to_numpy()
    masks = {
        "infant (<1y)": age < 1, "pediatric (<18y)": age < 18, "adult (18-64y)": (age >= 18) & (age < 65),
        "geriatric (65+y)": age >= 65, "very old (85+y)": age >= 85,
        "critical vitals missing": critical, "all critical vitals present": ~critical,
    }
    sex = df["sex"].astype(str).to_numpy()
    for label, code in (("female", "F"), ("male", "M")):
        if (sex == code).any():
            masks[label] = sex == code
    return masks


def subgroup_report(df: pd.DataFrame, pred_esi: np.ndarray, proba: Optional[np.ndarray]) -> Dict[str, Dict]:
    y = df["esi"].to_numpy()
    report = {}
    for name, m in subgroup_masks(df).items():
        if m.sum() == 0:
            continue
        row = evaluate(y[m], pred_esi[m], None if proba is None else proba[m])
        row["small_sample"] = bool(m.sum() < MIN_SUBGROUP_N)
        report[name] = row
    return report


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], Optional[float]],
                 n_boot: int = 500, seed: int = 0) -> Optional[List[float]]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = metric(y[idx], pred[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def macro_f1_metric(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0))


def high_acuity_sensitivity_metric(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    h = y <= 2
    return float((p[h] <= 2).mean()) if h.any() else None
