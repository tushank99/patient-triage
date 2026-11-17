"""Authentication, roles, per-patient authorization, access log, secrets."""
import pytest

from backend.tests.conftest import PASSWORD, auth, triage_payload

SCEN = {
    "mrn": "MRN-SIL-002", "age_years": 54.0, "gender": "M", "complaint": "Feels winded.",
    "vitals": {"heart_rate": 78, "systolic_bp": 122, "oxygen_saturation": 97.0},
}


def test_every_data_endpoint_requires_a_token(client):
    for method, path in [
        ("get", "/api/queue"), ("get", "/api/archive"), ("get", "/api/patients/MRN-A3F9"), ("get", "/api/auth/me"),
        ("post", "/api/triage"), ("post", "/api/override"), ("post", "/api/archive"), ("post", "/api/demo/surge"),
        ("get", "/api/visits/x/audit"), ("get", "/api/visits/x/vitals"),
    ]:
        r = getattr(client, method)(path)
        assert r.status_code == 401, (method, path, r.status_code)
        assert r.json()["error"]["code"] == "unauthenticated"
    assert client.get("/").status_code == 200  # health check stays public


def test_login_wrong_password_and_lockout(client):
    from backend.app.auth import clear_login_failures

    for _ in range(5):
        assert client.post("/api/auth/login", json={"username": "ghost", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "ghost", "password": "nope"}).status_code == 429
    clear_login_failures("ghost")
    assert client.post("/api/auth/login", json={"username": "lee", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "lee", "password": PASSWORD}).status_code == 200


def test_me_returns_role_and_permissions(client):
    me = client.get("/api/auth/me", headers=auth(client, "physician" if False else "patel")).json()
    assert me["role"] == "physician" and "override:below_floor" in me["permissions"]
    assert "override:below_floor" not in client.get("/api/auth/me", headers=auth(client, "lee")).json()["permissions"]


def test_tampered_and_expired_tokens_are_rejected(client):
    import jwt

    from backend.app.config import settings

    good = auth(client)["Authorization"].split()[1]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {good}x"}).status_code == 401
    forged = jwt.encode({"sub": "admin", "exp": 4102444800}, "some-other-secret-some-other-secret!!", algorithm="HS256")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    expired = jwt.encode({"sub": "lee", "exp": 1}, settings.jwt_secret, algorithm="HS256")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_viewer_sees_queue_without_names_and_nothing_else(client):
    client.post("/api/triage", json=triage_payload(SCEN), headers=auth(client))
    q = client.get("/api/queue", headers=auth(client, "viewer")).json()
    assert q and all(item["patient_name"] is None for item in q)
    assert any(item["patient_name"] for item in client.get("/api/queue", headers=auth(client, "lee")).json())
    for path in ("/api/patients/MRN-A3F9", "/api/archive"):
        assert client.get(path, headers=auth(client, "viewer")).status_code == 403
    assert client.post("/api/triage", json=triage_payload(SCEN), headers=auth(client, "viewer")).status_code == 403


def test_admin_has_no_patient_level_access(client):
    h = auth(client, "admin")
    assert client.get("/api/patients/MRN-A3F9", headers=h).status_code == 403
    assert client.post("/api/triage", json=triage_payload(SCEN), headers=h).status_code == 403
    assert all(i["patient_name"] is None for i in client.get("/api/queue", headers=h).json())


def test_other_facility_cannot_see_or_touch_a_visit(client):
    vid = client.post("/api/triage", json=triage_payload(SCEN), headers=auth(client, "lee")).json()["visit_id"]
    other = auth(client, "okoye")
    assert vid not in {q["visit_id"] for q in client.get("/api/queue", headers=other).json()}
    for method, path, body in [
        ("get", f"/api/visits/{vid}/audit", None), ("get", f"/api/visits/{vid}/vitals", None),
        ("post", f"/api/visits/{vid}/review", {}),
        ("post", "/api/override", {"visit_id": vid, "assigned_esi": 1, "override_reason": "snooping"}),
        ("post", "/api/archive", {"visit_id": vid, "disposition": "LWBS"}),
    ]:
        r = getattr(client, method)(path, headers=other, **({"json": body} if body is not None else {}))
        assert r.status_code == 404, (path, r.status_code)  # existence is not revealed
    from backend.app.database import SessionLocal
    from backend.app.models import AccessLog

    db = SessionLocal()
    denied = db.query(AccessLog).filter(AccessLog.username == "okoye", AccessLog.allowed.is_(False), AccessLog.visit_id == vid).count()
    db.close()
    assert denied >= 4


def test_override_below_rule_floor_needs_a_senior_role(client):
    scen = {**SCEN, "complaint": "Short of breath", "vitals": {**SCEN["vitals"], "oxygen_saturation": 88.0}}  # R-HYPOXIA -> floor ESI 2
    vid = client.post("/api/triage", json=triage_payload(scen), headers=auth(client, "lee")).json()["visit_id"]

    nurse = client.post("/api/override", json={"visit_id": vid, "assigned_esi": 4, "override_reason": "looks fine to me"}, headers=auth(client, "lee"))
    assert nurse.status_code == 403 and "charge nurse or physician" in nurse.json()["error"]["message"]
    assert client.post("/api/override", json={"visit_id": vid, "assigned_esi": 1, "override_reason": "worse"}, headers=auth(client, "lee")).status_code == 200
    assert client.post("/api/override", json={"visit_id": vid, "assigned_esi": 2, "override_reason": "at the floor"}, headers=auth(client, "lee")).json()["below_rule_floor"] is False

    ok = client.post("/api/override", json={"visit_id": vid, "assigned_esi": 4, "override_reason": "Reviewed: probe artefact, SpO2 98% on recheck"}, headers=auth(client, "patel"))
    assert ok.status_code == 200 and ok.json()["below_rule_floor"] is True
    last = client.get(f"/api/visits/{vid}/audit", headers=auth(client, "patel")).json()[-1]
    assert last["action"] == "OVERRIDE_ESI_BELOW_RULE_FLOOR" and last["actor"] == "patel"
    assert last["detail"]["rule_floor_esi"] == 2 and last["detail"]["role"] == "physician"


def test_patient_access_is_logged(client):
    from backend.app.database import SessionLocal
    from backend.app.models import AccessLog

    client.get("/api/patients/MRN-A3F9", headers=auth(client, "lee"))
    db = SessionLocal()
    row = db.query(AccessLog).filter(AccessLog.username == "lee", AccessLog.action == "PATIENT_LOOKUP").first()
    assert row and row.role == "triage_nurse" and row.patient_id and row.allowed
    assert db.query(AccessLog).filter(AccessLog.action == "LOGIN", AccessLog.username == "lee").count() >= 1
    db.close()


def test_passwords_are_hashed_and_verify():
    from backend.app.auth import hash_password, verify_password

    h = hash_password("correct horse battery")
    assert "correct" not in h and verify_password("correct horse battery", h) and not verify_password("wrong", h)
    assert hash_password("x") != hash_password("x")  # salted
    assert not verify_password("x", "garbage")


def test_protected_environments_refuse_weak_secrets(monkeypatch):
    from backend.app import config

    monkeypatch.setattr(config.settings, "environment", "production")
    monkeypatch.setattr(config.settings, "jwt_secret", "short")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        config.ensure_secrets()
    monkeypatch.setattr(config.settings, "jwt_secret", "x" * 40)
    config.ensure_secrets()


def test_demo_users_are_not_created_without_a_strong_password_in_protected_envs(monkeypatch):
    from backend.app import config
    from backend.scripts import seed_data

    class FakeDb:
        def query(self, *_):
            raise AssertionError("must not touch the DB")

    monkeypatch.setattr(config.settings, "environment", "production")
    monkeypatch.setattr(config.settings, "demo_user_password", "")
    assert seed_data.seed_users(FakeDb()) == 0
    monkeypatch.setattr(config.settings, "demo_user_password", "short")
    assert seed_data.seed_users(FakeDb()) == 0
