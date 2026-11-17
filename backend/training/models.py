"""
The three things we compare, plus calibration and explanations.

  rules-only       the deployed deterministic safety rules alone (floor ESI if any rule fires, else the most
                   common label among rows where none fired). The baseline any ML must justify itself against.
  logistic         standard multinomial logistic regression with median imputation + missingness indicators.
  xgboost          gradient-boosted trees on the shared feature schema (native NaN handling + *_missing flags).
  xgboost + rules  what actually ships: final = min(model ESI, rule floor). Rules can only raise acuity.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.clinical.rules_engine import evaluate_rules, rule_floor
from backend.app.ml.feature_schema import CLASS_TO_ESI, FEATURE_NAMES, MISSINGNESS_SOURCES, RAW_FEATURE_NAMES
from backend.app.ml.model import apply_temperature

N_CLASSES = len(CLASS_TO_ESI)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
    objective="multi:softprob", num_class=N_CLASSES, tree_method="hist", missing=np.nan, n_jobs=1,
)


# ------------------------------------------------------------------ features


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Same columns, same order, same derivation as ml/feature_schema.build_feature_vector."""
    x = df[RAW_FEATURE_NAMES].copy()
    for src in MISSINGNESS_SOURCES:
        x[f"{src}_missing"] = x[src].isna().astype(float)
    return x[FEATURE_NAMES].to_numpy(dtype=np.float32)


def raw_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[RAW_FEATURE_NAMES].to_numpy(dtype=np.float32)


def sqrt_balanced_weights(y_class: np.ndarray) -> np.ndarray:
    """Rare, dangerous classes (ESI 1) get more weight, softened by sqrt so the model is not distorted."""
    counts = np.bincount(y_class, minlength=N_CLASSES).astype(float)
    w = np.where(counts > 0, (counts.sum() / (N_CLASSES * np.maximum(counts, 1))) ** 0.5, 0.0)
    return w[y_class]


def full_proba(model, x: np.ndarray) -> np.ndarray:
    """predict_proba as (n, 5) even if a class was absent from this training fold."""
    p = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = model.steps[-1][1].classes_
    if classes is None or len(classes) == N_CLASSES:
        return p
    out = np.zeros((len(x), N_CLASSES))
    out[:, np.asarray(classes, dtype=int)] = p
    return out


# ------------------------------------------------------------------ models


def fit_logreg(x: np.ndarray, y_class: np.ndarray):
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    pipe.fit(x, y_class, logisticregression__sample_weight=sqrt_balanced_weights(y_class))
    return pipe


def fit_xgb(x: np.ndarray, y_class: np.ndarray, seed: int = 42):
    from xgboost import XGBClassifier

    present = np.unique(y_class)
    if len(present) != N_CLASSES:
        raise ValueError(
            f"Training data has no examples of ESI {sorted(set(range(N_CLASSES)) - set(present.tolist()))} "
            f"(classes present: {[CLASS_TO_ESI[c] for c in present]}). Use a larger dataset."
        )
    model = XGBClassifier(random_state=seed, **XGB_PARAMS)
    model.fit(x, y_class, sample_weight=sqrt_balanced_weights(y_class))
    return model


class RulesOnly:
    """Predicts the safety-rule floor when any rule fires, else `default_esi` (learned from training labels)."""

    def __init__(self) -> None:
        self.default_esi = 3

    @staticmethod
    def floors(df: pd.DataFrame) -> np.ndarray:
        """Rule floor per row (NaN when no rule fires). Training data has no structured history, so the
        history-dependent anticoagulant rule cannot fire here; it still applies in production."""
        cols = RAW_FEATURE_NAMES
        out = np.full(len(df), np.nan)
        records = df[cols + ["complaint"]].to_dict("records")
        for i, r in enumerate(records):
            inputs = {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "complaint"}
            inputs["complaint"] = r["complaint"]
            f = rule_floor(evaluate_rules(inputs))
            if f is not None:
                out[i] = f
        return out

    def fit(self, df: pd.DataFrame, floors: Optional[np.ndarray] = None) -> "RulesOnly":
        floors = self.floors(df) if floors is None else floors
        rest = df["esi"].to_numpy()[np.isnan(floors)]
        self.default_esi = int(pd.Series(rest).mode().iloc[0]) if len(rest) else 3
        return self

    def predict(self, df: pd.DataFrame, floors: Optional[np.ndarray] = None) -> np.ndarray:
        floors = self.floors(df) if floors is None else floors
        return np.where(np.isnan(floors), self.default_esi, floors).astype(int)


def combine_with_rules(model_esi: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """The deployed decision: rules can only raise acuity (smaller ESI number)."""
    return np.where(np.isnan(floors), model_esi, np.minimum(model_esi, np.nan_to_num(floors, nan=9))).astype(int)


# ------------------------------------------------------------------ calibration


def fit_temperature(proba: np.ndarray, y_class: np.ndarray) -> float:
    """One scalar T minimising NLL on out-of-fold probabilities. Cannot change the argmax."""
    eps = 1e-12
    yc = np.asarray(y_class)

    def nll(t: float) -> float:
        q = apply_temperature(proba, t)
        return float(-np.log(np.clip(q[np.arange(len(yc)), yc], eps, 1)).mean())

    res = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded")
    return round(float(res.x), 4)


# ------------------------------------------------------------------ explanations (TreeSHAP)


def shap_values(model, x: np.ndarray, predicted_class: np.ndarray) -> np.ndarray:
    """(n, features) TreeSHAP of the predicted class, from XGBoost's own pred_contribs (no `shap` package)."""
    import xgboost as xgb

    contribs = model.get_booster().predict(xgb.DMatrix(x, missing=np.nan), pred_contribs=True)  # (n, classes, feats+1)
    return contribs[np.arange(len(x)), predicted_class, :-1]


def global_importance(model, x: np.ndarray, predicted_class: np.ndarray, top: int = 15) -> List[Tuple[str, float]]:
    sv = np.abs(shap_values(model, x, predicted_class)).mean(axis=0)
    order = np.argsort(-sv)[:top]
    return [(FEATURE_NAMES[i], round(float(sv[i]), 4)) for i in order]
