"""
Train a CANDIDATE model. Nothing here touches the shipped artifacts; `promote.py` does that, through the gate.

  python -m backend.training.train --dataset synthetic                       # plumbing / demo only
  python -m backend.training.train --dataset mimic --path <mimic-iv root>    # needs credentialed PhysioNet access
  python -m backend.training.train --dataset ktas  --path ktas.csv           # public Korean data (KTAS != ESI)
  python -m backend.training.train --dataset mimic --path <root> --inspect   # data-quality report only

Pipeline: locked test split -> grouped CV on development data (rules-only vs logistic vs XGBoost) ->
temperature calibration from out-of-fold predictions -> final fit -> ONE evaluation on the untouched test
set (overall + pediatric/geriatric/missing-vitals subgroups + bootstrap CIs) -> TreeSHAP importance ->
ONNX export from the same fitted model -> parity check -> model card + report.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.feature_schema import (  # noqa: E402
    CLASS_TO_ESI, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FEATURES, RAW_FEATURE_NAMES, build_feature_vector, schema_as_dict,
)
from backend.app.ml.model import TriageModel, apply_temperature  # noqa: E402
from backend.training import metrics as M  # noqa: E402
from backend.training.datasets import Dataset, load  # noqa: E402
from backend.training.models import (  # noqa: E402
    XGB_PARAMS, RulesOnly, combine_with_rules, feature_matrix, fit_logreg, fit_temperature, fit_xgb, full_proba,
    global_importance, raw_matrix,
)
from backend.training.splits import cv_folds, locked_split  # noqa: E402
from backend.training.tracking import get_tracker, git_sha  # noqa: E402

RUNS_DIR = ROOT / "backend" / "ml_runs"
ESI = np.array(CLASS_TO_ESI)
MODEL_FAMILY = "xgb-v3"


def clean(obj: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _decide(proba: np.ndarray) -> np.ndarray:
    return ESI[proba.argmax(axis=1)]


def _summarise(folds: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in ("macro_f1", "high_acuity_sensitivity", "severe_under_triage", "ece", "nll"):
        vals = [f[key] for f in folds if f.get(key) is not None]
        if vals:
            out[key] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    return out


def _cv(dev, x_dev, raw_dev, floors_dev, k: int, seed: int, ablate: List[str]) -> Dict[str, Any]:
    y = dev["esi"].to_numpy()
    yc = y - 1
    oof_xgb, oof_lr = np.zeros((len(dev), 5)), np.zeros((len(dev), 5))
    pred_rules = np.zeros(len(dev), dtype=int)
    per_fold: Dict[str, List[Dict]] = {m: [] for m in ("rules_only", "logreg", "xgboost", "xgboost+rules")}
    for tr, va in cv_folds(dev, k, seed):
        ro = RulesOnly().fit(dev.iloc[tr], floors_dev[tr])
        pred_rules[va] = ro.predict(dev.iloc[va], floors_dev[va])
        oof_lr[va] = full_proba(fit_logreg(raw_dev[tr], yc[tr]), raw_dev[va])
        oof_xgb[va] = full_proba(fit_xgb(x_dev[tr], yc[tr], seed), x_dev[va])
        pred_x = _decide(oof_xgb[va])
        per_fold["rules_only"].append(M.evaluate(y[va], pred_rules[va]))
        per_fold["logreg"].append(M.evaluate(y[va], _decide(oof_lr[va]), oof_lr[va]))
        per_fold["xgboost"].append(M.evaluate(y[va], pred_x, oof_xgb[va]))
        per_fold["xgboost+rules"].append(M.evaluate(y[va], combine_with_rules(pred_x, floors_dev[va]), oof_xgb[va]))
    pooled_x = _decide(oof_xgb)
    result = {
        "folds": k,
        "pooled": {
            "rules_only": M.evaluate(y, pred_rules), "logreg": M.evaluate(y, _decide(oof_lr), oof_lr),
            "xgboost": M.evaluate(y, pooled_x, oof_xgb), "xgboost+rules": M.evaluate(y, combine_with_rules(pooled_x, floors_dev), oof_xgb),
        },
        "fold_mean_std": {m: _summarise(v) for m, v in per_fold.items()},
    }
    # Ablations: does the model lean on a feature that will mean something different in production?
    ablations = {}
    for name in ablate:
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n != name]
        oof = np.zeros((len(dev), 5))
        for tr, va in cv_folds(dev, k, seed):
            oof[va] = full_proba(fit_xgb(x_dev[tr][:, keep], yc[tr], seed), x_dev[va][:, keep])
        ablations[f"without {name}"] = M.evaluate(y, combine_with_rules(_decide(oof), floors_dev), oof)
    result["ablations"] = ablations
    result["oof_xgb"] = oof_xgb  # popped by the caller
    return result


def _parity(model: TriageModel, onnx_path: str, n: int = 200, seed: int = 7) -> Dict[str, Any]:
    """Server-side proof that the exported ONNX graph + calibration reproduces the joblib model."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    ranges = {  # realistic spread incl. abnormal values, so the trees' extreme branches are exercised
        "heart_rate": (35, 190), "systolic_bp": (60, 220), "diastolic_bp": (35, 120), "respiratory_rate": (6, 40),
        "oxygen_saturation": (75, 100), "temperature_celsius": (35.0, 40.5), "pain_score": (0, 10),
    }
    ints = {"heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "pain_score"}
    vectors: List[Dict] = []
    for _ in range(n):
        row: Dict[str, Any] = {"age_years": round(float(rng.uniform(0.05, 95)), 2), "has_prior_history": bool(rng.random() < 0.5)}
        for name, (lo, hi) in ranges.items():
            if rng.random() < 0.25:
                row[name] = None
            else:
                v = rng.uniform(lo, hi)
                row[name] = int(round(v)) if name in ints else round(float(v), 1)
        vectors.append(row)
    x = np.array([build_feature_vector(v) for v in vectors], dtype=np.float32)
    expected = apply_temperature(full_proba(model.model, x), model.temperature)
    sess = ort.InferenceSession(onnx_path)
    got_raw = sess.run(None, {sess.get_inputs()[0].name: x})
    outputs = dict(zip([o.name for o in sess.get_outputs()], got_raw))
    got = apply_temperature(np.asarray(outputs["probabilities"], dtype=np.float64), model.temperature)
    mismatches = int((ESI[got.argmax(1)] != ESI[expected.argmax(1)]).sum())
    return {
        "n": n, "max_abs_prob_diff": float(np.abs(got - expected).max()), "esi_mismatches": mismatches,
        "vectors": [{"input": v, "probabilities": [round(float(p), 6) for p in expected[i]], "esi": int(ESI[expected[i].argmax()])}
                    for i, v in enumerate(vectors)],
    }


def _export_onnx(xgb_model, path: str) -> None:
    import onnx
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx.save_model(convert_xgboost(xgb_model, initial_types=[("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]), path)


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _limitations(label_source: str) -> List[str]:
    common = [
        "Calibration measures agreement with the label source, not clinical truth.",
        "ESI labels are nurse judgements at triage and are themselves noisy (inter-rater agreement is moderate).",
        "has_prior_history is 'earlier ED stay in this dataset' in training but 'history documents on file' in production.",
        "No prospective or multi-site validation. Not a medical device. Deterministic safety rules always apply on top.",
    ]
    if label_source == "synthetic-rules":
        return ["SYNTHETIC LABELS: generated by if/else rules on the same features, so the model mostly re-learns that recipe and every "
                "metric measures agreement with the recipe, not with clinicians. Plumbing and demo use only."] + common
    if label_source == "ktas-korea":
        return ["KTAS is a different triage scale from ESI (similar five levels, different criteria); single country; small sample."] + common
    return ["MIMIC-IV-ED is single-site (Beth Israel Deaconess) and covers 2011-2019; first stays per patient are censored for has_prior_history."] + common


def run_training(ds: Dataset, seed: int = 42, folds: int = 5, ablate: Optional[List[str]] = None, use_mlflow: bool = True,
                 runs_dir: Path = RUNS_DIR, external: Optional[Dataset] = None, n_boot: int = 500, tag: str = "") -> Dict[str, Any]:
    ablate = ["has_prior_history"] if ablate is None else ablate
    df = ds.df
    dev_idx, test_idx, split = locked_split(df, 0.2, seed)
    dev, test = df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    x_dev, raw_dev, yc_dev = feature_matrix(dev), raw_matrix(dev), dev["esi"].to_numpy() - 1
    floors_dev = RulesOnly.floors(dev)

    cv = _cv(dev, x_dev, raw_dev, floors_dev, folds, seed, ablate)
    oof_xgb = cv.pop("oof_xgb")
    temperature = fit_temperature(oof_xgb, yc_dev)

    # ---- final fit on all development data, then the single look at the test set
    lr, xgb_model = fit_logreg(raw_dev, yc_dev), fit_xgb(x_dev, yc_dev, seed)
    x_test, raw_test, y_test = feature_matrix(test), raw_matrix(test), test["esi"].to_numpy()
    floors_test = RulesOnly.floors(test)
    p_raw = full_proba(xgb_model, x_test)
    p_cal = apply_temperature(p_raw, temperature)
    p_lr = full_proba(lr, raw_test)
    pred_x, pred_lr = _decide(p_cal), _decide(p_lr)
    pred_rules = RulesOnly().fit(dev, floors_dev).predict(test, floors_test)
    pred_combo = combine_with_rules(pred_x, floors_test)

    yc_test = y_test - 1
    test_res: Dict[str, Any] = {
        "rules_only": {"overall": M.evaluate(y_test, pred_rules), "subgroups": M.subgroup_report(test, pred_rules, None)},
        "logreg": {"overall": M.evaluate(y_test, pred_lr, p_lr), "subgroups": M.subgroup_report(test, pred_lr, p_lr)},
        "xgb_raw": {"overall": M.evaluate(y_test, _decide(p_raw), p_raw)},
        "xgb_calibrated": {"overall": M.evaluate(y_test, pred_x, p_cal), "temperature": temperature,
                           "reliability_before": M.reliability_bins(p_raw, yc_test), "reliability_after": M.reliability_bins(p_cal, yc_test)},
        "xgb_rules": {"overall": M.evaluate(y_test, pred_combo, p_cal), "subgroups": M.subgroup_report(test, pred_combo, p_cal),
                      "ci95": {"macro_f1": M.bootstrap_ci(y_test, pred_combo, M.macro_f1_metric, n_boot),
                               "high_acuity_sensitivity": M.bootstrap_ci(y_test, pred_combo, M.high_acuity_sensitivity_metric, n_boot)}},
    }
    importance = global_importance(xgb_model, x_test, p_cal.argmax(axis=1))

    external_res = None
    if external is not None:
        ex = external.df
        ex_floors = RulesOnly.floors(ex)
        ex_p = apply_temperature(full_proba(xgb_model, feature_matrix(ex)), temperature)
        ex_pred = combine_with_rules(_decide(ex_p), ex_floors)
        external_res = {"dataset": external.name, "label_source": external.label_source, "caveat": external.info.get("caveat"),
                        "overall": M.evaluate(ex["esi"].to_numpy(), ex_pred, ex_p), "subgroups": M.subgroup_report(ex, ex_pred, ex_p)}

    # ---- run folder, export, parity
    created = datetime.now(timezone.utc)
    run_id = f"{created:%Y%m%d-%H%M%S}-{ds.label_source}-{ds.sha256[:6]}{('-' + tag) if tag else ''}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(xgb_model, run_dir / "triage_model.joblib")
    _export_onnx(xgb_model, str(run_dir / "triage_model.onnx"))
    version = f"{MODEL_FAMILY}-{ds.label_source}-{created:%Y%m%d}"
    meta = {
        "model_version": version, "label_source": ds.label_source, "run_id": run_id, **schema_as_dict(),
        "calibration": {"method": "temperature", "temperature": temperature, "fit_on": "out-of-fold development predictions"},
        "joblib_sha256": _sha(str(run_dir / "triage_model.joblib")), "onnx_sha256": _sha(str(run_dir / "triage_model.onnx")),
        "onnx_file": "triage_model.onnx",
    }
    (run_dir / "triage_model.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    parity = _parity(TriageModel(xgb_model, meta), str(run_dir / "triage_model.onnx"))
    vectors = parity.pop("vectors")
    (run_dir / "parity_vectors.json").write_text(json.dumps({"model_version": version, "temperature": temperature, "vectors": vectors}), encoding="utf-8")

    # ---- test-set accounting: every extra evaluation on the same locked test set is recorded
    usage_path = runs_dir / "test_set_usage.json"
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
    key = f"{ds.sha256[:16]}:{split['test_groups_sha256'][:16]}"
    usage.setdefault(key, []).append(run_id)
    usage_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")

    card = clean({
        "run_id": run_id, "created_at": created.isoformat(), "model_version": version, "label_source": ds.label_source,
        "dataset": {"name": ds.name, "info": ds.info}, "dataset_sha256": ds.sha256, "split": split, "seed": seed, "git_sha": git_sha(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "features": FEATURE_NAMES, "raw_features": RAW_FEATURE_NAMES,
        "xgb_params": {**XGB_PARAMS, "missing": "NaN"}, "calibration": meta["calibration"], "cv": cv,
        "test": test_res, "test_set": {"n_evaluations": len(usage[key]), "run_ids": usage[key]},
        "external": external_res, "importance": importance, "parity": parity,
        "limitations": _limitations(ds.label_source),
    })
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(card), encoding="utf-8")

    tracker = get_tracker(str(run_dir), str(runs_dir), use_mlflow)
    tracker.start(run_id, {"seed": seed, "folds": folds, "xgb": card["xgb_params"], "temperature": temperature},
                  {"label_source": ds.label_source, "dataset_sha256": ds.sha256, "dataset_name": ds.name, "git_sha": git_sha() or "",
                   "model_version": version, "test_groups_sha256": split["test_groups_sha256"], "feature_schema": FEATURE_SCHEMA_VERSION})
    tracker.log_metrics({"test": {m: test_res[m]["overall"] for m in ("rules_only", "logreg", "xgb_raw", "xgb_calibrated", "xgb_rules")},
                         "parity": {"max_abs_prob_diff": parity["max_abs_prob_diff"], "esi_mismatches": parity["esi_mismatches"]},
                         "cv_pooled": cv["pooled"], "test_set_evaluations": card["test_set"]["n_evaluations"]})
    tracker.log_artifacts(str(run_dir))
    tracker.end()
    card["tracker_backend"] = tracker.backend
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "card": card}


# ------------------------------------------------------------------ report


def _row(name: str, m: Dict) -> str:
    f = lambda k: "n/a" if m.get(k) is None else f"{m[k]:.3f}"  # noqa: E731
    return f"| {name} | {m['n']} | {f('macro_f1')} | {f('qwk')} | {f('high_acuity_sensitivity')} | {f('under_triage')} | {f('severe_under_triage')} | {f('ece')} | {f('nll')} |"


def render_report(card: Dict) -> str:
    head = "| model | n | macro-F1 | QWK | ESI1-2 sensitivity | under-triage | severe under | ECE | NLL |\n|---|---|---|---|---|---|---|---|---|"
    t = card["test"]
    lines = [f"# Model report: {card['model_version']}", ""]
    if card["label_source"] == "synthetic-rules":
        lines += ["> **SYNTHETIC LABELS. These numbers are NOT evidence of clinical performance.** The labels come from rules on the same",
                  "> features, so every model is graded against the recipe that generated its labels, not against clinicians.", ""]
    lines += [f"* run `{card['run_id']}` · git `{card.get('git_sha')}` · feature schema v{card['feature_schema_version']}",
              f"* dataset `{card['dataset']['name']}` ({card['label_source']}) sha256 `{card['dataset_sha256'][:16]}` · {card['dataset']['info'].get('n_rows')} rows",
              f"* locked test set: {card['split']['n_test']} rows / {card['split']['n_test_groups']} patients · **evaluated {card['test_set']['n_evaluations']}x** "
              "(each extra look makes the test set less independent)",
              f"* calibration: temperature {card['calibration']['temperature']} (fit on out-of-fold development predictions)", "",
              "## Test set (untouched until the final evaluation)", head]
    for key, name in (("rules_only", "rules-only"), ("logreg", "logistic regression"), ("xgb_raw", "XGBoost (uncalibrated)"),
                      ("xgb_calibrated", "XGBoost (calibrated)"), ("xgb_rules", "**XGBoost + rules (ships)**")):
        lines.append(_row(name, t[key]["overall"]))
    ci = t["xgb_rules"]["ci95"]
    lines += ["", f"95% bootstrap CI for the shipped combination: macro-F1 {ci['macro_f1']}, ESI 1-2 sensitivity {ci['high_acuity_sensitivity']}", "",
              "## Subgroups (shipped combination)", head]
    for name, m in t["xgb_rules"]["subgroups"].items():
        lines.append(_row(name + (" (small n)" if m.get("small_sample") else ""), m))
    lines += ["", "## Cross-validation (development data, grouped by patient)", head]
    for name, m in card["cv"]["pooled"].items():
        lines.append(_row(name, m))
    for name, m in card["cv"].get("ablations", {}).items():
        lines.append(_row(f"xgboost+rules, {name}", m))
    lines += ["", "## Calibration (test set, reliability bins: mean confidence vs accuracy)", "", "before:"]
    lines += [f"* {b['bin']}: n={b['n']}, confidence {b['mean_confidence']}, accuracy {b['accuracy']}" for b in t["xgb_calibrated"]["reliability_before"]]
    lines += ["", "after:"] + [f"* {b['bin']}: n={b['n']}, confidence {b['mean_confidence']}, accuracy {b['accuracy']}" for b in t["xgb_calibrated"]["reliability_after"]]
    lines += ["", "## What drives the model (mean |TreeSHAP|, predicted class, log-odds)"] + [f"* {n}: {v}" for n, v in card["importance"]]
    if card.get("external"):
        e = card["external"]
        lines += ["", f"## External check: {e['dataset']} ({e['label_source']})", f"_{e.get('caveat') or ''}_", head, _row("shipped combination", e["overall"])]
    p = card["parity"]
    lines += ["", f"## ONNX parity: max |dp| {p['max_abs_prob_diff']:.2e}, {p['esi_mismatches']} ESI mismatches over {p['n']} vectors", "", "## Limitations"]
    lines += [f"* {x}" for x in card["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["synthetic", "mimic", "ktas"], required=True)
    ap.add_argument("--path")
    ap.add_argument("--external-dataset", choices=["ktas", "mimic"])
    ap.add_argument("--external-path")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-synthetic", type=int, default=8000)
    ap.add_argument("--inspect", action="store_true", help="print the data-quality report and stop")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ds = load(a.dataset, a.path, **({"n": a.n_synthetic, "seed": a.seed} if a.dataset == "synthetic" else {}))
    if a.inspect:
        print(json.dumps(clean({"name": ds.name, "label_source": ds.label_source, "sha256": ds.sha256, **ds.info}), indent=2))
        return
    ext = load(a.external_dataset, a.external_path) if a.external_dataset else None
    out = run_training(ds, a.seed, a.folds, use_mlflow=not a.no_mlflow, external=ext, tag=a.tag)
    print(f"Candidate written to {out['run_dir']}\nReport: {out['run_dir']}/report.md  (tracker: {out['card'].get('tracker_backend')})")
    print(f"Next: python -m backend.training.promote --run {out['card']['run_id']} --channel demo|research|production")


if __name__ == "__main__":
    main()
