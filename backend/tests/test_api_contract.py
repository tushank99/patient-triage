"""API contract, idempotency and workflow tests."""
import json
import uuid

from backend.tests.conftest import REPO_ROOT, auth, triage_payload

OPENAPI_SNAPSHOT = REPO_ROOT / "contracts" / "openapi.json"

BASE = {
    "mrn": "MRN-SIL-002", "age_years": 54.0, "gender": "M", "complaint": "Feels winded.",
    "vitals": {"heart_rate": 78, "systolic_bp": 122, "diastolic_bp": 78, "respiratory_rate": 16,
               "oxygen_saturation": 97.0, "temperature_celsius": 36.8, "pain_score": 0},
}


def test_openapi_snapshot_is_current(client):
    """Changing a schema without regenerating contracts/openapi.json (and the frontend types) fails here.
    Fix: python -m backend.scripts.export_openapi && (cd frontend && npm run gen:api)"""
    from backend.app.main import app

    assert OPENAPI_SNAPSHOT.exists(), "run python -m backend.scripts.export_openapi"
    committed = json.loads(OPENAPI_SNAPSHOT.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(app.openapi())), "API contract changed; regenerate the snapshot and frontend types"


def test_old_update_vitals_payload_is_rejected_with_error_envelope(client):
    old = {"mrn": "MRN-C2M8", "name": "Reyes, Miguel", "age": 58, "complaint": "SOB", "hr": 110, "spo2": 88}
    r = client.post("/api/triage", json=old, headers=auth(client))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    locs = {tuple(d["loc"]) for d in err["details"]}
    assert ("body", "request_id") in locs and ("body", "hr") in locs


def test_clinician_identity_cannot_be_supplied_by_the_client(client):
    """The actor comes from the token. A body field claiming another identity is rejected outright."""
    r = client.post("/api/override", json={"visit_id": "x", "clinician_id": "dr.house", "assigned_esi": 1, "override_reason": "because"}, headers=auth(client))
    assert r.status_code == 422


def test_retry_with_same_request_id_replays_result(client):
    body = triage_payload(BASE)
    first = client.post("/api/triage", json=body, headers=auth(client))
    second = client.post("/api/triage", json=body, headers=auth(client))
    assert first.status_code == second.status_code == 200 and first.json() == second.json()


def test_same_request_id_different_payload_conflicts(client):
    body = triage_payload(BASE)
    assert client.post("/api/triage", json=body, headers=auth(client)).status_code == 200
    body["chief_complaint"] = "Something else"
    r = client.post("/api/triage", json=body, headers=auth(client))
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"


def test_two_walk_ins_back_to_back_do_not_block_each_other(client):
    a = client.post("/api/triage", json=triage_payload({**BASE, "mrn": None}), headers=auth(client))
    b = client.post("/api/triage", json=triage_payload({**BASE, "mrn": None}), headers=auth(client))
    assert a.status_code == b.status_code == 200 and a.json()["mrn"] != b.json()["mrn"]
    assert a.json()["is_new_patient"] and a.json()["mrn"].startswith("WALK-")


def test_unknown_mrn_is_404_not_a_silent_new_patient(client):
    r = client.post("/api/triage", json=triage_payload({**BASE, "mrn": "MRN-DOES-NOT-EXIST"}), headers=auth(client))
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_demographic_mismatch_warns(client):
    assert client.post("/api/triage", json=triage_payload({**BASE, "gender": "F"}), headers=auth(client)).json()["warnings"]


def test_patient_lookup_computes_has_history(client):
    rich = client.get("/api/patients/MRN-A3F9", headers=auth(client)).json()
    assert rich["has_history"] is True and rich["document_count"] >= 5
    none = client.get("/api/patients/MRN-ZER-004", headers=auth(client)).json()
    assert none["has_history"] is False and none["document_count"] == 0
    assert client.get("/api/patients/NOPE", headers=auth(client)).status_code == 404


def test_reassess_escalates_immediately_downgrades_need_confirmation_and_vitals_history_is_kept(client):
    created = client.post("/api/triage", json=triage_payload(BASE), headers=auth(client)).json()
    vid = created["visit_id"]
    h = auth(client)

    worse = client.post(f"/api/visits/{vid}/reassess", json={"request_id": str(uuid.uuid4()), "vitals": {"oxygen_saturation": 88.0}}, headers=h).json()
    assert worse["final_esi"] <= 2 and worse["requires_confirmation"] is False

    better = client.post(f"/api/visits/{vid}/reassess", json={"request_id": str(uuid.uuid4()), "vitals": {"oxygen_saturation": 99.0}}, headers=h).json()
    if better["recommended_esi"] > worse["final_esi"]:
        assert better["requires_confirmation"] is True and better["final_esi"] == worse["final_esi"]

    ok = client.post("/api/override", json={"visit_id": vid, "assigned_esi": 3, "override_reason": "Improved on O2"}, headers=h)
    assert ok.status_code == 200 and ok.json()["final_esi"] == 3

    readings = client.get(f"/api/visits/{vid}/vitals", headers=h).json()
    assert [r["source"] for r in readings] == ["INTAKE", "REASSESS", "REASSESS"]
    assert readings[1]["oxygen_saturation"] == 88.0 and readings[1]["heart_rate"] is None  # only what was measured

    queue = {q["visit_id"]: q for q in client.get("/api/queue", headers=h).json()}
    assert queue[vid]["esi_score"] == 3 and queue[vid]["is_overridden"] is True
    actions = [e["action"] for e in client.get(f"/api/visits/{vid}/audit", headers=h).json()]
    assert actions == ["AI_TRIAGE", "AI_REASSESS", "AI_REASSESS", "OVERRIDE_ESI"]


def test_review_flag_can_be_acknowledged(client):
    vid = client.post("/api/triage", json=triage_payload(BASE), headers=auth(client)).json()["visit_id"]
    q = {x["visit_id"]: x for x in client.get("/api/queue", headers=auth(client)).json()}
    assert q[vid]["needs_review"] is True and q[vid]["review_reasons"]
    assert client.post(f"/api/visits/{vid}/review", json={"note": "seen"}, headers=auth(client)).status_code == 200
    q = {x["visit_id"]: x for x in client.get("/api/queue", headers=auth(client)).json()}
    assert q[vid]["needs_review"] is False


def test_needs_reassessment_patient_can_be_dispositioned_and_reverted(client):
    from backend.app.database import SessionLocal
    from backend.app.models import Visit

    h = auth(client)
    vid = client.post("/api/triage", json=triage_payload(BASE), headers=h).json()["visit_id"]
    db = SessionLocal()
    db.get(Visit, vid).status = "NEEDS_REASSESSMENT"
    db.commit()
    db.close()

    assert client.post("/api/archive", json={"visit_id": vid, "disposition": "ADMITTED", "checklist": {"paged": True}}, headers=h).status_code == 200
    archived = {a["visit_id"]: a for a in client.get("/api/archive", headers=h).json()}
    assert archived[vid]["disposition"] == "ADMITTED" and archived[vid]["patient_name"]
    assert client.post("/api/archive/revert", json={"visit_id": vid}, headers=h).status_code == 200


def test_monitor_flags_overdue_visits_and_audits_it(client):
    from datetime import datetime, timedelta, timezone

    from backend.app.background_tasks import flag_overdue_visits
    from backend.app.database import SessionLocal
    from backend.app.models import AuditLog, Visit

    h = auth(client)
    vid = client.post("/api/triage", json=triage_payload({**BASE, "mrn": None}), headers=h).json()["visit_id"]
    assert flag_overdue_visits(datetime.now(timezone.utc) + timedelta(days=2)) >= 1
    db = SessionLocal()
    assert db.get(Visit, vid).status == "NEEDS_REASSESSMENT"
    assert db.query(AuditLog).filter(AuditLog.visit_id == vid, AuditLog.action == "DETERIORATION_FLAGGED").count() == 1
    db.close()


def test_surge_is_non_destructive_idempotent_and_role_protected(client):
    from backend.app.models import Visit  # noqa: F401

    assert client.post("/api/demo/surge", headers=auth(client, "lee")).status_code == 403          # triage nurse
    base_count = len(client.get("/api/queue", headers=auth(client, "rivera")).json())
    r = client.post("/api/demo/surge", headers=auth(client, "rivera"))
    assert r.status_code == 200 and r.json()["visits_changed"] == 10
    q = client.get("/api/queue", headers=auth(client, "rivera")).json()
    assert len(q) == base_count + 10 and sum(x["is_demo_surge"] for x in q) == 10
    assert client.post("/api/demo/surge", headers=auth(client, "rivera")).json()["visits_changed"] == 0  # idempotent
    assert client.post("/api/demo/surge/end", headers=auth(client, "rivera")).json()["visits_changed"] == 10
    assert len(client.get("/api/queue", headers=auth(client, "rivera")).json()) == base_count
    assert client.get("/api/archive", headers=auth(client, "rivera")).status_code == 200
    assert client.post("/api/simulate-surge").status_code in (404, 405)


def test_anonymize_is_admin_only_and_deletes_history_documents(client):
    from backend.app.database import SessionLocal
    from backend.app.models import ClinicalDocument, Patient

    db = SessionLocal()
    p = Patient(mrn="MRN-TEST-ANON", name="Test, Person", age_years=40, gender="F")
    db.add(p)
    db.flush()
    db.add(ClinicalDocument(id="MRN-TEST-ANON:note", patient_id=p.id, doc_type="er_note", title="t", content="c", checksum="x", source="test"))
    db.commit()
    db.close()

    assert client.post("/api/archive/anonymize", json={"mrn": "MRN-TEST-ANON"}, headers=auth(client, "rivera")).status_code == 403
    assert client.post("/api/archive/anonymize", json={"mrn": "MRN-TEST-ANON"}, headers=auth(client, "admin")).status_code == 200
    db = SessionLocal()
    assert db.query(ClinicalDocument).filter(ClinicalDocument.id == "MRN-TEST-ANON:note").count() == 0
    db.close()
    assert client.get("/api/patients/MRN-TEST-ANON", headers=auth(client)).status_code == 404
