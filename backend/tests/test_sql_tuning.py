"""Guards for the SQL tuning work in docs/sql-tuning.md (partial indexes on active visits, queue N+1 fix)."""
import re
import uuid

from sqlalchemy import event

from backend.tests.conftest import auth


def test_partial_index_predicate_matches_active_statuses():
    """If ACTIVE_STATUSES changes, the partial indexes must change with it, or the queue silently goes back to seq scans."""
    from backend.app.models import ACTIVE_STATUSES, Visit

    for idx in Visit.__table__.indexes:
        if idx.name.startswith("ix_visits_active_"):
            for dialect in ("postgresql", "sqlite"):
                predicate = str(idx.dialect_options[dialect]["where"])
                assert set(re.findall(r"'([A-Z_]+)'", predicate)) == set(ACTIVE_STATUSES), (idx.name, dialect)


def test_queue_loads_patients_in_the_same_query(client):
    """The waiting room used to issue one extra SELECT per row to lazy-load each visit's patient (N+1)."""
    from backend.app.database import engine

    h = auth(client, "rivera")
    for i in range(3):
        body = {"request_id": str(uuid.uuid4()), "age_years": 40 + i, "gender": "F", "patient_name": f"N+1 check {i}",
                "chief_complaint": "Mild ankle pain after a fall.",
                "vitals": {"heart_rate": 80, "systolic_bp": 120, "diastolic_bp": 80, "respiratory_rate": 16,
                           "oxygen_saturation": 99.0, "temperature_celsius": 36.8, "pain_score": 3}}
        assert client.post("/api/triage", json=body, headers=h).status_code == 200

    lazy_loads = []

    def spy(conn, cursor, statement, *_):
        # SQLAlchemy's lazy-load of Visit.patient: a patients-only SELECT by primary key (the queue query itself JOINs)
        if re.search(r"FROM patients\s+WHERE patients\.id =", statement) and "JOIN" not in statement:
            lazy_loads.append(statement)

    event.listen(engine, "before_cursor_execute", spy)
    try:
        r = client.get("/api/queue", headers=h)
    finally:
        event.remove(engine, "before_cursor_execute", spy)
    assert r.status_code == 200 and len(r.json()) >= 3
    assert lazy_loads == [], f"{len(lazy_loads)} per-row patient queries: the queue regressed to N+1"
