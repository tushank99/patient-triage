"""Safety order, de-identification and structured fallbacks, tested end to end with a fake LLM."""
from backend.app.deps import get_llm_fn
from backend.app.llm.assessor import LLMUnavailable
from backend.app.main import app
from backend.tests.conftest import auth, fake_llm, triage_payload

SILENT_HYPOXIA = {
    "mrn": "MRN-SIL-002", "age_years": 54.0, "gender": "M",
    "complaint": "Feels slightly winded after walking up stairs. Denies chest pain.",
    "vitals": {"heart_rate": 78, "systolic_bp": 122, "diastolic_bp": 78, "respiratory_rate": 26,
               "oxygen_saturation": 89.0, "temperature_celsius": 36.8, "pain_score": 0},
}


def post(client, payload):
    return client.post("/api/triage", json=payload, headers=auth(client)).json()


def test_llm_can_never_downgrade_a_rules_flagged_case(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=4)
    body = post(client, triage_payload(SILENT_HYPOXIA))
    assert body["components"]["llm_esi"] == 4
    assert body["recommended_esi"] == 2 and body["components"]["rule_floor_esi"] == 2
    assert body["components"]["llm_downgrade_blocked"]
    assert body["rationale"].startswith("SAFETY RULE ESCALATION")
    assert any("overrode the AI" in r for r in body["review_reasons"]) and body["requires_clinician_review"]
    stages = [s["stage"] for s in body["decision_trace"]]
    assert stages[0] == "base" and stages[-1] == "rule_floor"


def test_llm_can_escalate_above_the_floor(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=1)
    assert post(client, triage_payload(SILENT_HYPOXIA))["recommended_esi"] == 1


def test_llm_failure_falls_back_to_ml_plus_rules(client):
    def broken(_):
        raise LLMUnavailable("llm_unavailable")

    app.dependency_overrides[get_llm_fn] = lambda: broken
    body = post(client, triage_payload(SILENT_HYPOXIA))
    assert "llm_unavailable" in body["degraded"] and body["components"]["llm_esi"] is None
    assert body["recommended_esi"] <= 2 and body["requires_clinician_review"]


def test_invented_citations_are_dropped_and_refs_are_mapped_to_real_ids(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=2, cited=["D1", "D99", "MRN-SIL-002:problem_list"])
    body = post(client, triage_payload(SILENT_HYPOXIA))
    ids = {d["id"] for d in body["history"]["documents"]}
    assert body["history"]["cited_document_ids"] and set(body["history"]["cited_document_ids"]) <= ids
    assert len(body["history"]["cited_document_ids"]) == 1  # D99 and the raw id were rejected


def test_nothing_identifying_reaches_the_llm(client):
    captured = []
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=2, captured=captured)
    payload = triage_payload(
        {**SILENT_HYPOXIA, "mrn": "MRN-A3F9", "age_years": 45, "gender": "F"},
        patient_name="Jane, Anne",
        chief_complaint="Anne Jane says she does feel winded since 2025-03-04, call 555-123-4567 or a.jane@example.com",
    )
    body = post(client, payload)
    sent = captured[0].complaint + captured[0].history_block
    for leaked in ("Anne", "Jane", "555-123-4567", "example.com", "MRN-A3F9", "2025-03-04", "2022-03", "2024-10"):
        assert leaked not in sent, leaked
    assert "does feel winded" in sent                      # old scrubber redacted 'Doe' inside 'does'
    assert "[D1]" in captured[0].history_block             # opaque refs, not document ids
    assert "ago" in captured[0].history_block               # dates became relative
    assert body["pii_redactions"] >= 4


def test_history_reaches_llm_with_temporal_markers(client):
    captured = []
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=2, captured=captured)
    post(client, triage_payload({**SILENT_HYPOXIA, "mrn": "MRN-A3F9", "age_years": 45, "gender": "F", "complaint": "Crushing chest pain"}))
    block = captured[0].history_block
    assert "Penicillin" in block and "RESOLVED" in block and "HISTORICAL" in block


def test_age_over_89_is_aggregated_for_the_llm(client):
    captured = []
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=3, captured=captured)
    post(client, triage_payload({**SILENT_HYPOXIA, "mrn": None, "age_years": 93.0}))
    assert captured[0].age_label == "90+"


def test_confidence_is_three_separate_signals(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=2)
    body = post(client, triage_payload(SILENT_HYPOXIA))
    c = body["confidence"]
    assert set(c) >= {"model_probability", "data_completeness", "retrieval_confidence"}
    assert 0 <= c["model_probability"] <= 1 and 0 <= c["data_completeness"] <= 1 and 0 <= c["retrieval_confidence"] <= 1
    assert "confidence_score" not in body  # the single blended number is gone


def test_walk_in_has_no_retrieval_confidence_and_missing_critical_vitals_force_review(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=3, category="RESPIRATORY")
    body = post(client, triage_payload({**SILENT_HYPOXIA, "mrn": None, "vitals": {"heart_rate": 90}}))
    assert body["confidence"]["retrieval_confidence"] is None
    assert body["confidence"]["data_completeness"] <= 0.5
    assert "SpO2" in body["confidence"]["missing_critical"] and body["requires_clinician_review"]


def test_audit_trail_records_what_was_sent_and_decided(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=2, risk_factors=["asthma history"])
    body = post(client, triage_payload(SILENT_HYPOXIA))
    events = client.get(f"/api/visits/{body['visit_id']}/audit", headers=auth(client)).json()
    ai = next(e for e in events if e["action"] == "AI_TRIAGE")
    snap = ai["detail"]
    assert snap["llm"]["request_text"] and snap["llm"]["provider"] == "fake" and snap["inputs"]["vitals"]
    assert snap["assessment"]["versions"]["rules"] and snap["assessment"]["decision_trace"]
    visit_row = next(q for q in client.get("/api/queue", headers=auth(client)).json() if q["visit_id"] == body["visit_id"])
    assert visit_row["ai_model_probability"] is not None
    from backend.app.database import SessionLocal
    from backend.app.models import Visit

    db = SessionLocal()
    v = db.get(Visit, body["visit_id"])
    assert v.ai_risk_factors and "asthma history" in v.ai_risk_factors and v.ai_assessment
    db.close()
