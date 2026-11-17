"""Phase 2: the ML pipeline. Fast unit tests plus one end-to-end train -> gate -> promote run on synthetic data."""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.feature_schema import FEATURE_NAMES, RAW_FEATURE_NAMES, build_feature_vector
from backend.app.ml.model import apply_temperature
from backend.training import metrics as M
from backend.training.datasets import load_ktas, load_mimic_ed, load_synthetic
from backend.training.gate import evaluate_gate
from backend.training.models import RulesOnly, combine_with_rules, feature_matrix, fit_temperature
from backend.training.splits import cv_folds, locked_split
from backend.tests.conftest import REPO_ROOT

DEMO = REPO_ROOT / "backend" / "data" / "external" / "mimic-iv-ed-demo"


# ------------------------------------------------------------------ features


def test_derived_missingness_indicators_match_between_training_and_serving():
    df = load_synthetic(300, seed=1).df.head(60)
    x = feature_matrix(df)
    for i, (_, row) in enumerate(df.iterrows()):
        served = np.array(build_feature_vector({k: (None if pd.isna(row[k]) else row[k]) for k in RAW_FEATURE_NAMES}), dtype=np.float32)
        np.testing.assert_array_equal(np.nan_to_num(x[i], nan=-1), np.nan_to_num(served, nan=-1))
    assert "oxygen_saturation_missing" in FEATURE_NAMES and "respiratory_rate" in FEATURE_NAMES


# ------------------------------------------------------------------ adapters


def _gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_mimic_adapter_cleans_units_text_and_history(tmp_path):
    _gz(tmp_path / "ed/triage.csv.gz",
        "subject_id,stay_id,temperature,heartrate,resprate,o2sat,sbp,dbp,pain,acuity,chiefcomplaint\n"
        "1,10,98.6,80,16,97,120,80,unable,3,CHEST PAIN\n"
        "1,11,101.3,300,16,999,120,80,13,2,FEVER\n"          # hr 300 ok edge, spo2 999 impossible, pain 13 invalid
        "2,12,,,,,,,,,TRAUMA\n"                               # no label -> dropped
        "3,13,97.0,70,14,99,110,70,5,4,SORE THROAT\n")        # subject 3 has no patients row -> no age -> dropped
    _gz(tmp_path / "ed/edstays.csv.gz",
        "subject_id,hadm_id,stay_id,intime,outtime,gender,race,arrival_transport,disposition\n"
        "1,,10,2150-01-01 10:00:00,2150-01-01 12:00:00,F,X,WALK IN,HOME\n"
        "1,,11,2150-06-01 10:00:00,2150-06-01 12:00:00,F,X,WALK IN,HOME\n"
        "2,,12,2150-02-01 10:00:00,2150-02-01 12:00:00,M,X,AMBULANCE,ADMITTED\n"
        "3,,13,2150-03-01 10:00:00,2150-03-01 12:00:00,M,X,WALK IN,HOME\n")
    _gz(tmp_path / "hosp/patients.csv.gz", "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod\n1,F,60,2148,2011 - 2013,\n2,M,40,2150,2011 - 2013,\n")
    ds = load_mimic_ed(str(tmp_path))
    df = ds.df.sort_values("heart_rate").reset_index(drop=True)
    assert ds.label_source == "mimic-iv-ed" and len(df) == 2
    first, second = df.iloc[0], df.iloc[1]
    assert first["temperature_celsius"] == pytest.approx(37.0, abs=0.01)          # 98.6 F -> 37 C
    assert np.isnan(first["pain_score"]) and first["age_years"] == 62               # 'unable' -> NaN; 60 + (2150-2148)
    assert first["has_prior_history"] == 0.0 and second["has_prior_history"] == 1.0  # second stay has history
    assert np.isnan(second["oxygen_saturation"]) and np.isnan(second["pain_score"])  # impossible values become missing, row kept
    assert second["temperature_celsius"] == pytest.approx(38.5, abs=0.01)
    assert ds.info["n_dropped_no_label_or_age"] == 2 and ds.info["n_out_of_range_set_missing"]["oxygen_saturation"] == 1
    assert df["group"].tolist() == [1, 1]  # grouped by patient


def test_ktas_adapter_handles_odd_values_and_reports_missing_columns(tmp_path):
    p = tmp_path / "ktas.csv"
    p.write_text("Sex;Age;SBP;DBP;HR;RR;BT;Saturation;NRS_pain;KTAS_expert;Chief_complain\n"
                 "1;70;120;80;88;18;36.5;97;3;3;chest pain\n"
                 "2;30;??;70;??;16;37.2;99;#BOL!;4;sprain\n", encoding="latin-1")
    ds = load_ktas(str(p))
    assert ds.label_source == "ktas-korea" and len(ds.df) == 2
    assert ds.df["sex"].tolist() == ["F", "M"] and ds.df["esi"].tolist() == [3, 4]
    assert ds.df.loc[1, ["systolic_bp", "heart_rate", "pain_score"]].isna().all()
    bad = tmp_path / "bad.csv"
    bad.write_text("a;b\n1;2\n")
    with pytest.raises(ValueError, match="lacks columns"):
        load_ktas(str(bad))


@pytest.mark.skipif(not (DEMO / "ed" / "triage.csv.gz").exists(), reason="open MIMIC-IV-ED demo not downloaded")
def test_mimic_adapter_on_the_real_open_demo_files():
    ds = load_mimic_ed(str(DEMO))
    assert ds.label_source == "mimic-iv-ed" and len(ds.df) > 100
    assert ds.df["temperature_celsius"].dropna().between(30, 45).all()  # converted from Fahrenheit
    assert ds.df["pain_score"].dropna().between(0, 10).all()            # 'unable', 'UA', 13 handled
    assert set(ds.df["esi"].unique()) <= {1, 2, 3, 4, 5}


# ------------------------------------------------------------------ splits


def test_splits_never_share_a_patient_and_are_reproducible():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"esi": rng.integers(1, 6, n), "group": rng.integers(0, 150, n)})  # ~4 visits per patient
    dev, test, manifest = locked_split(df, 0.2, seed=3)
    assert not set(df.iloc[dev]["group"]) & set(df.iloc[test]["group"])
    assert locked_split(df, 0.2, seed=3)[2] == manifest and locked_split(df, 0.2, seed=4)[2]["test_groups_sha256"] != manifest["test_groups_sha256"]
    devdf = df.iloc[dev].reset_index(drop=True)
    for tr, va in cv_folds(devdf, 5, 3):
        assert not set(devdf.iloc[tr]["group"]) & set(devdf.iloc[va]["group"])


# ------------------------------------------------------------------ metrics, calibration


def test_triage_metrics_count_the_right_kind_of_error():
    y = np.array([1, 2, 3, 3, 4, 5])
    p = np.array([2, 2, 3, 5, 3, 5])   # ESI1->2 (still high acuity), 3->5 severe under, 4->3 over
    m = M.evaluate(y, p)
    assert m["under_triage"] == pytest.approx(2 / 6, abs=1e-3) and m["severe_under_triage"] == pytest.approx(1 / 6, abs=1e-3) and m["over_triage"] == pytest.approx(1 / 6, abs=1e-3)
    assert m["high_acuity_sensitivity"] == 1.0
    assert M.evaluate(y, np.array([3, 2, 3, 3, 4, 5]))["high_acuity_sensitivity"] == 0.5  # missed one ESI 1
    assert M.evaluate(y, y)["macro_f1"] == 1.0


def test_ece_is_zero_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.3, 0.99, n)
    correct = rng.random(n) < conf
    proba = np.full((n, 5), 0.0)
    y = np.zeros(n, dtype=int)
    proba[:, 0] = conf
    proba[:, 1] = 1 - conf
    y[~correct] = 1
    assert M.expected_calibration_error(proba, y) < 0.02
    assert M.expected_calibration_error(np.where(proba > 0.5, 0.99, 0.01), y) > 0.1


def test_temperature_scaling_fixes_overconfidence_without_changing_decisions():
    rng = np.random.default_rng(1)
    n = 6000
    true_p = rng.dirichlet(np.ones(5) * 0.7, n)
    y = np.array([rng.choice(5, p=p) for p in true_p])
    overconfident = apply_temperature(true_p, 0.4)
    t = fit_temperature(overconfident, y)
    assert 2.0 < t < 3.0   # recovers ~1/0.4
    nll = lambda q: -np.log(q[np.arange(n), y]).mean()  # noqa: E731
    assert nll(apply_temperature(overconfident, t)) < nll(overconfident)
    assert (apply_temperature(overconfident, t).argmax(1) == overconfident.argmax(1)).all()
    np.testing.assert_allclose(apply_temperature(true_p, 1.0), true_p)


def test_combining_with_rules_can_only_raise_acuity():
    rng = np.random.default_rng(2)
    model = rng.integers(1, 6, 500)
    floors = np.where(rng.random(500) < 0.4, rng.integers(1, 4, 500), np.nan)
    out = combine_with_rules(model, floors)
    assert (out <= model).all()
    assert (out[~np.isnan(floors)] <= floors[~np.isnan(floors)]).all()
    assert (out[np.isnan(floors)] == model[np.isnan(floors)]).all()


def test_rules_only_baseline_uses_the_deployed_rules():
    df = pd.DataFrame({
        "age_years": [40.0, 40.0], "heart_rate": [80.0, 80.0], "systolic_bp": [120.0, 120.0], "diastolic_bp": [80.0, 80.0],
        "respiratory_rate": [16.0, 16.0], "oxygen_saturation": [88.0, 99.0], "temperature_celsius": [37.0, 37.0],
        "pain_score": [2.0, 2.0], "has_prior_history": [0.0, 0.0], "complaint": ["", ""], "esi": [2, 4],
    })
    floors = RulesOnly.floors(df)
    assert floors[0] == 2 and np.isnan(floors[1])
    assert RulesOnly().fit(df, floors).predict(df, floors).tolist() == [2, 4]


# ------------------------------------------------------------------ gate


def _card(**over):
    m = {"n": 500, "macro_f1": 0.70, "high_acuity_sensitivity": 0.92, "severe_under_triage": 0.01, "ece": 0.03, "high_acuity_n": 80}
    sub = {"pediatric (<18y)": {**m, "n": 60, "high_acuity_sensitivity": 0.90}, "geriatric (65+y)": {**m, "n": 90, "high_acuity_sensitivity": 0.91}}
    card = {
        "label_source": "mimic-iv-ed", "dataset_sha256": "abc", "parity": {"max_abs_prob_diff": 1e-7, "esi_mismatches": 0, "n": 200},
        "test": {"rules_only": {"overall": {**m, "macro_f1": 0.4, "high_acuity_sensitivity": 0.7}}, "logreg": {"overall": {**m, "macro_f1": 0.6}},
                 "xgb_calibrated": {"overall": m}, "xgb_rules": {"overall": m, "subgroups": sub}},
    }
    card.update(over)
    return card


GOLDEN_OK = {"n": 21, "hits": 17, "exact_match": 17 / 21, "under_triaged": 0, "baseline": 0.76}


def _failed(result):
    return {c.name for c in result.checks if c.status == "fail" and c.blocking}


def test_gate_passes_a_good_real_data_candidate():
    assert evaluate_gate(_card(), None, "production", "Dr X", GOLDEN_OK).passed


def test_synthetic_labels_can_ship_only_to_the_demo_channel():
    synthetic = _card(label_source="synthetic-rules")
    assert evaluate_gate(synthetic, None, "demo", None, GOLDEN_OK).passed
    for channel in ("research", "production"):
        assert "label source allowed" in _failed(evaluate_gate(synthetic, None, channel, "Dr X", GOLDEN_OK))


def test_production_needs_a_named_signoff_and_ktas_cannot_reach_it():
    assert "clinical sign-off recorded" in _failed(evaluate_gate(_card(), None, "production", None, GOLDEN_OK))
    assert "label source allowed" in _failed(evaluate_gate(_card(label_source="ktas-korea"), None, "production", "Dr X", GOLDEN_OK))
    assert evaluate_gate(_card(label_source="ktas-korea"), None, "research", None, GOLDEN_OK).passed


def test_gate_blocks_regressions_parity_golden_and_subgroup_gaps():
    shipped = _card()
    worse = _card()
    worse["test"]["xgb_rules"]["overall"] = {**worse["test"]["xgb_rules"]["overall"], "high_acuity_sensitivity": 0.88}
    assert "high-acuity sensitivity" in _failed(evaluate_gate(worse, shipped, "demo", None, GOLDEN_OK))

    assert "ONNX vs joblib" in _failed(evaluate_gate(_card(parity={"max_abs_prob_diff": 0.01, "esi_mismatches": 0, "n": 200}), None, "demo", None, GOLDEN_OK))
    assert "golden exact-match" in _failed(evaluate_gate(_card(), None, "demo", None, {**GOLDEN_OK, "exact_match": 0.5, "hits": 10}))
    assert "no golden under-triage" in _failed(evaluate_gate(_card(), None, "demo", None, {**GOLDEN_OK, "under_triaged": 1}))

    gap = _card()
    gap["test"]["xgb_rules"]["subgroups"]["pediatric (<18y)"]["high_acuity_sensitivity"] = 0.70
    assert "pediatric (<18y)" in _failed(evaluate_gate(gap, None, "research", None, GOLDEN_OK))
    assert evaluate_gate(gap, None, "demo", None, GOLDEN_OK).passed  # demo channel only warns on subgroup gaps

    weaker_than_rules = _card()
    weaker_than_rules["test"]["xgb_rules"]["overall"] = {**weaker_than_rules["test"]["xgb_rules"]["overall"], "high_acuity_sensitivity": 0.65}
    assert "not worse than rules-only (high-acuity)" in _failed(evaluate_gate(weaker_than_rules, None, "demo", None, GOLDEN_OK))


def test_regression_is_skipped_when_the_dataset_changed():
    other = _card(dataset_sha256="different")
    other["test"]["xgb_rules"]["overall"] = {**other["test"]["xgb_rules"]["overall"], "macro_f1": 0.3}
    res = evaluate_gate(other, _card(), "demo", None, GOLDEN_OK)
    assert any(c.group == "regression" and c.status == "skip" for c in res.checks)


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from backend.training.train import run_training

    runs = tmp_path_factory.mktemp("runs")
    out = run_training(load_synthetic(1800, seed=5), seed=5, folds=3, use_mlflow=False, runs_dir=runs, n_boot=40)
    return {"runs": runs, **out}


def test_training_run_produces_a_complete_honest_card(trained):
    card, run = trained["card"], Path(trained["run_dir"])
    for f in ("triage_model.joblib", "triage_model.onnx", "triage_model.meta.json", "model_card.json", "report.md", "parity_vectors.json"):
        assert (run / f).exists(), f
    assert card["label_source"] == "synthetic-rules" and card["limitations"][0].startswith("SYNTHETIC LABELS")
    assert (run / "report.md").read_text(encoding="utf-8").count("NOT evidence of clinical performance") == 1
    t = card["test"]
    assert set(t) >= {"rules_only", "logreg", "xgb_raw", "xgb_calibrated", "xgb_rules"}
    assert {"pediatric (<18y)", "geriatric (65+y)", "critical vitals missing"} <= set(t["xgb_rules"]["subgroups"])
    assert t["xgb_rules"]["ci95"]["macro_f1"] and card["calibration"]["temperature"] > 0
    assert "without has_prior_history" in card["cv"]["ablations"]
    assert card["parity"]["max_abs_prob_diff"] < 1e-4 and card["parity"]["esi_mismatches"] == 0
    # the shipped decision can only be at least as sensitive to high-acuity patients as the model alone
    assert t["xgb_rules"]["overall"]["high_acuity_sensitivity"] >= t["xgb_raw"]["overall"]["high_acuity_sensitivity"] - 1e-9


def test_test_set_reuse_is_counted(trained):
    from backend.training.train import run_training

    again = run_training(load_synthetic(1800, seed=5), seed=5, folds=3, use_mlflow=False, runs_dir=trained["runs"], n_boot=10, ablate=[], tag="again")
    assert again["card"]["test_set"]["n_evaluations"] == 2


def test_treeshap_contributions_sum_to_the_models_margin(trained):
    import joblib
    import xgboost as xgb

    model = joblib.load(Path(trained["run_dir"]) / "triage_model.joblib")
    x = feature_matrix(load_synthetic(200, seed=9).df)[:25]
    d = xgb.DMatrix(x, missing=np.nan)
    contribs = model.get_booster().predict(d, pred_contribs=True)
    margin = model.get_booster().predict(d, output_margin=True)
    np.testing.assert_allclose(contribs.sum(axis=-1), margin, atol=1e-3)


def test_promotion_goes_through_the_gate(trained, tmp_path, monkeypatch):
    from backend.training.promote import promote

    monkeypatch.setattr("backend.training.promote.golden_regression", lambda model: GOLDEN_OK)  # golden check is covered by gate tests
    kw = dict(runs_dir=trained["runs"], artifact_dir=tmp_path / "art", frontend_dir=tmp_path / "web", card_path=tmp_path / "card.json")
    run_id = trained["card"]["run_id"]
    assert promote(run_id, "production", "Dr X", **kw)["passed"] is False      # synthetic labels never reach production
    assert not (tmp_path / "card.json").exists()
    assert promote(run_id, "demo", dry_run=True, **kw)["copied"] is False
    out = promote(run_id, "demo", **kw)
    assert out["passed"] and out["copied"]
    shipped = json.loads((tmp_path / "card.json").read_text())
    assert shipped["promoted"]["channel"] == "demo" and shipped["label_source"] == "synthetic-rules"
    for f in ("triage_model.onnx", "triage_model.meta.json", "parity_vectors.json", "safety_rules.json"):
        assert (tmp_path / "web" / f).exists()
    assert (tmp_path / "art" / "triage_model.joblib").exists()
