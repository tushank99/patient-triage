"""
Replays TriageFlow's real database access patterns against a Postgres loaded by generate_load.sql, so
pg_stat_statements traces exactly the SQL the app sends (same SQLAlchemy ORM expressions as the app code).

  DATABASE_URL=postgresql://postgres:tuning@localhost:5433/triage \
      python -m backend.scripts.sql_tuning.workload [--rounds 300] [--no-reset]

LOCAL ONLY. Every query below cites the app code it mirrors; keep them in sync if that code changes.
"""
import argparse
import random
import time
from collections import defaultdict

from sqlalchemy import text

from backend.app.database import SessionLocal
from backend.app.models import ACTIVE_STATUSES, ClinicalDocument, Patient, Visit
from backend.app.services.history_store import count_documents


def history_rows(db, patient):  # services/history_store.py retrieve_history()
    return db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == patient.id, ClinicalDocument.status == "READY").all()


def document_list(db, patient):  # routers/documents.py list_documents()
    q = db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == patient.id, ClinicalDocument.status != "DELETED")
    return q.order_by(ClinicalDocument.doc_date.desc().nullslast(), ClinicalDocument.created_at.desc()).all()


def duplicate_check(db, patient):  # ingestion/service.py upload: same-checksum document for this patient
    return (db.query(ClinicalDocument)
            .filter(ClinicalDocument.patient_id == patient.id, ClinicalDocument.checksum == "no-such-checksum",
                    ClinicalDocument.status != "DELETED").first())


def waiting_room(db, facility):  # routers/queue.py get_waiting_room()
    return (db.query(Visit).join(Patient)
            .filter(Visit.status.in_(ACTIVE_STATUSES), Visit.facility_id == facility)
            .order_by(Visit.final_esi_score.asc(), Visit.arrival_time.asc()).all())


def deterioration_monitor(db):  # background_tasks.py flag_overdue_visits()
    return db.query(Visit).filter(Visit.status == "WAITING").all()


def active_visit_check(db, patient):  # ops/retention.py: skip anonymising a patient who is still in the queue
    return db.query(Visit).filter(Visit.patient_id == patient.id, Visit.status.in_(ACTIVE_STATUSES)).count()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--no-reset", action="store_true", help="keep existing pg_stat_statements counters")
    args = ap.parse_args()

    random.seed(7)
    db = SessionLocal()
    if not args.no_reset:
        db.execute(text("SELECT pg_stat_statements_reset()"))
        db.commit()
    patients = [Patient(id=f"pat-{random.randint(1, 50000)}") for _ in range(args.rounds)]
    timings = defaultdict(float)

    def run(name, fn, *a):
        t = time.perf_counter()
        fn(db, *a)
        timings[name] += time.perf_counter() - t
        db.rollback()  # no long transactions; also drops identity-map caching between calls

    for i, p in enumerate(patients):
        run("count_documents", count_documents, p)  # every triage
        run("history_rows", history_rows, p)  # every triage
        run("document_list", document_list, p)  # chart view
        run("duplicate_check", duplicate_check, p)  # every upload
        run("waiting_room", waiting_room, "ED-MAIN" if i % 4 else "ED-NORTH")  # dashboards poll this
        run("active_visit_check", active_visit_check, p)  # retention job, per patient
        if i % 3 == 0:
            run("deterioration_monitor", deterioration_monitor)  # background tick
    db.close()

    print(f"{'query':24} {'total ms':>10} {'mean ms':>9}")
    for name, total in sorted(timings.items(), key=lambda kv: -kv[1]):
        n = len(patients) if name != "deterioration_monitor" else (len(patients) + 2) // 3
        print(f"{name:24} {total * 1000:10.1f} {total * 1000 / n:9.2f}")


if __name__ == "__main__":
    main()
