import hashlib
import json
import re

import numpy as np
import pytest

from backend.tests.conftest import REPO_ROOT

FRONTEND_MODELS = REPO_ROOT / "frontend" / "public" / "models"
ARTIFACTS = REPO_ROOT / "backend" / "app" / "ml" / "artifacts"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- seed / master set


def test_seed_is_idempotent(seeded):
    assert seeded["users_created"] == 6
    from backend.scripts.seed_data import seed

    again = seed(users=True, queue=True)
    assert again["patients_created"] == again["docs_created"] == again["docs_updated"] == again["users_created"] == 0
    assert again["queue_visits"] == 11
    assert seed(users=True, queue=True)["queue_visits"] == 0  # re-running never duplicates visits
    assert again["docs_unchanged"] == seeded["docs_created"]


def test_master_set_history_mix():
    from backend.scripts.seed_data import load_master, render_documents

    patients = load_master()["patients"]
    profiles = [p["profile"] for p in patients]
    assert len({p["mrn"] for p in patients}) == len(patients)
    assert 0.70 <= profiles.count("rich") / len(patients) <= 0.80
    assert 3 <= profiles.count("none") <= 4
    assert profiles.count("conflicting") >= 2
    for p in patients:
        assert (len(render_documents(p)) == 0) == (p["profile"] == "none")


def test_demo_queue_uses_only_master_set_patients():
    from backend.app.services.demo_queue import load_demo_queue
    from backend.scripts.seed_data import load_master

    master = {p["mrn"] for p in load_master()["patients"]}
    q = load_demo_queue()
    assert {e["mrn"] for g in ("base", "surge") for e in q[g]} <= master
    assert not {e["mrn"] for e in q["base"]} & {e["mrn"] for e in q["surge"]}


def test_frontend_has_no_hardcoded_patient_lists():
    """The backend owns the queue; the UI must not carry its own demo patients."""
    ctx = (REPO_ROOT / "frontend" / "src" / "lib" / "triage-context.tsx").read_text(encoding="utf-8")
    assert "BASE_PATIENTS" not in ctx and "SURGE_PATIENTS" not in ctx and "MRN-" not in ctx


# ---------------------------------------------------------------- model bundle


def test_frontend_bundle_matches_backend_artifacts():
    """Fails if someone retrains or edits rules without re-running `promote`."""
    meta = json.loads((ARTIFACTS / "triage_model.meta.json").read_text(encoding="utf-8"))
    assert (FRONTEND_MODELS / "triage_model.meta.json").read_bytes() == (ARTIFACTS / "triage_model.meta.json").read_bytes()
    assert (FRONTEND_MODELS / "safety_rules.json").read_bytes() == (REPO_ROOT / "backend" / "app" / "clinical" / "safety_rules.json").read_bytes()
    assert _sha(FRONTEND_MODELS / "triage_model.onnx") == meta["onnx_sha256"] == _sha(ARTIFACTS / "triage_model.onnx")
    assert _sha(ARTIFACTS / "triage_model.joblib") == meta["joblib_sha256"]
    assert json.loads((FRONTEND_MODELS / "parity_vectors.json").read_text())["model_version"] == meta["model_version"]


def test_shipped_model_card_is_honest_about_its_labels():
    from backend.app.ml.model import get_model, get_model_card

    card, model = get_model_card(), get_model()
    assert card and model and card["model_version"] == model.version and card["label_source"] == model.label_source
    assert card["promoted"]["gate"]["passed"] is True and card["parity"]["esi_mismatches"] == 0
    if card["label_source"] == "synthetic-rules":
        assert card["promoted"]["channel"] == "demo", "synthetic-label models may only ship to the demo channel"


def test_model_meta_matches_feature_schema():
    from backend.app.ml.feature_schema import CLASS_TO_ESI, FEATURE_NAMES
    from backend.app.ml.model import get_model

    model = get_model()
    assert model is not None
    assert [f["name"] for f in model.meta["features"]] == FEATURE_NAMES
    assert model.meta["class_to_esi"] == CLASS_TO_ESI and model.meta["calibration"]["temperature"] > 0
    assert model.model.n_features_in_ == len(FEATURE_NAMES)


def test_onnx_bundle_reproduces_the_server_model_on_the_shared_vectors():
    """Python half of the parity proof (the browser half is frontend/scripts/verify-onnx-parity.mjs)."""
    ort = pytest.importorskip("onnxruntime")
    from backend.app.ml.feature_schema import build_feature_vector
    from backend.app.ml.model import apply_temperature, get_model

    model = get_model()
    vectors = json.loads((FRONTEND_MODELS / "parity_vectors.json").read_text())["vectors"]
    x = np.array([build_feature_vector(v["input"]) for v in vectors], dtype=np.float32)
    sess = ort.InferenceSession(str(FRONTEND_MODELS / "triage_model.onnx"))
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {sess.get_inputs()[0].name: x})))
    got = apply_temperature(np.asarray(out["probabilities"], dtype=np.float64), model.temperature)
    np.testing.assert_allclose(got, np.array([v["probabilities"] for v in vectors]), atol=1e-4)
    assert [int(model.meta["class_to_esi"][i]) for i in got.argmax(1)] == [v["esi"] for v in vectors]
    # The raw ONNX "label" output is a CLASS INDEX (0 = ESI 1); decoding must go through class_to_esi.
    assert (out["label"] == got.argmax(axis=1)).all()


def test_runtime_model_returns_calibrated_probabilities_and_exact_explanations():
    from backend.app.ml.model import get_model

    model = get_model()
    res = model.predict({"age_years": 54.0, "heart_rate": 78, "systolic_bp": 122, "respiratory_rate": 26, "oxygen_saturation": 89.0, "has_prior_history": True})
    assert abs(sum(res.probabilities) - 1) < 1e-6 and res.top_probability == max(res.probabilities)
    assert 1 <= len(res.contributions) <= 4 and all(c.label for c in res.contributions)
    assert any(c.feature in ("oxygen_saturation", "respiratory_rate", "temperature_celsius_missing", "diastolic_bp") for c in res.contributions)
