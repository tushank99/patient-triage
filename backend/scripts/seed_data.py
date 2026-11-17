"""
Idempotent seed: loads the synthetic patient master set into the database.

  python -m backend.scripts.seed_data                  # patients + history documents (safe to re-run, never deletes)
  python -m backend.scripts.seed_data --users --queue  # also demo accounts and the demo waiting room
  python -m backend.scripts.seed_data --reset          # LOCAL ONLY: drop + recreate all tables first

- Patients are upserted by MRN.
- Documents are rendered from each patient's structured facts, so the facts are ground truth.
- Each document has a stable id and a sha256 checksum; unchanged documents are skipped.
- Demo users (--users) and the demo waiting room (--queue) are optional and idempotent too.

Replaces the old script that deleted every patient, visit and audit log.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.database import SessionLocal  # noqa: E402
from backend.app.db_migrate import reset_database, run_migrations  # noqa: E402
from backend.app.auth import hash_password  # noqa: E402
from backend.app.ingestion.service import ingest_text_inline  # noqa: E402
from backend.app.models import Patient, User  # noqa: E402

logger = logging.getLogger(__name__)

DEMO_USERS = [
    ("rivera", "Sarah Rivera, RN", "charge_nurse", "ED-MAIN"),
    ("lee", "Jordan Lee, RN", "triage_nurse", "ED-MAIN"),
    ("patel", "Dr. Anita Patel", "physician", "ED-MAIN"),
    ("admin", "System Administrator", "admin", "ED-MAIN"),
    ("viewer", "Waiting Room Display", "viewer", "ED-MAIN"),
    ("okoye", "Chris Okoye, RN (other facility)", "triage_nurse", "ED-NORTH"),
]
DEV_DEMO_PASSWORD = "demo-password-1"  # only used outside staging/production when DEMO_USER_PASSWORD is unset

MASTER_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "patient_master.json")


@dataclass
class RenderedDoc:
    id: str
    doc_type: str
    title: str
    doc_date: Optional[date]
    content: str

    @property
    def checksum(self) -> str:
        payload = json.dumps([self.doc_type, self.title, str(self.doc_date), self.content])
        return hashlib.sha256(payload.encode()).hexdigest()


def load_master() -> dict:
    with open(MASTER_PATH, encoding="utf-8") as f:
        return json.load(f)


def age_on(dob: date, today: date) -> float:
    return round((today - dob).days / 365.25, 2)


def _latest_date(p: dict) -> date:
    dates = [e["date"] for e in p.get("encounters", [])]
    return date.fromisoformat(max(dates)) if dates else date(2026, 1, 1)


def render_documents(p: dict) -> List[RenderedDoc]:
    """Turn structured facts into the documents the retriever indexes."""
    if p["profile"] == "none":
        return []
    mrn, docs = p["mrn"], []
    as_of = _latest_date(p)

    if p.get("conditions"):
        lines = []
        for c in p["conditions"]:
            if c["status"] == "resolved":
                lines.append(f"- {c['name']}: RESOLVED ({c['onset']} to {c.get('resolved', '?')})")
            elif c["status"] == "historical":
                lines.append(f"- {c['name']}: HISTORICAL event ({c['onset']})")
            else:
                lines.append(f"- {c['name']}: ACTIVE since {c['onset']}")
        docs.append(RenderedDoc(f"{mrn}:problem_list", "problem_list", "Problem list", as_of, "Problem list:\n" + "\n".join(lines)))

    if p.get("medications") or p["profile"] == "rich":
        lines = []
        for m in p.get("medications", []):
            cls = m["class"].replace("_", " ")
            if m["status"] == "active":
                lines.append(f"- {m['name']} {m['dose']} ({cls}): ACTIVE since {m['since']}")
            else:
                lines.append(f"- {m['name']} {m['dose']} ({cls}): DISCONTINUED {m.get('stopped', '')}".rstrip())
        body = "\n".join(lines) if lines else "- No regular medications."
        docs.append(RenderedDoc(f"{mrn}:medication_list", "medication_list", "Medication list", as_of, "Medication list:\n" + body))

    if p.get("allergies") or p["profile"] == "rich":
        lines = [f"- {a['substance']}: {a['reaction']} (severity: {a['severity']})" for a in p.get("allergies", [])]
        body = "\n".join(lines) if lines else "- No known drug allergies (NKDA)."
        docs.append(RenderedDoc(f"{mrn}:allergy_record", "allergy_record", "Allergy record", as_of, "Allergies:\n" + body))

    for e in p.get("encounters", []):
        docs.append(RenderedDoc(
            f"{mrn}:{e['doc_type']}:{e['date']}", e["doc_type"], e["title"], date.fromisoformat(e["date"]),
            f"[DATE: {e['date']}] {e['title']}. {e['text']}",
        ))
    return docs


def history_facts(p: dict) -> Optional[dict]:
    if p["profile"] == "none":
        return None
    return {k: p.get(k, []) for k in ("conditions", "medications", "allergies")}


def seed_users(db) -> int:
    """Demo accounts. Password comes from DEMO_USER_PASSWORD; protected environments refuse a default."""
    password = settings.demo_user_password
    if not password:
        if settings.is_protected_env:
            logger.warning("DEMO_USER_PASSWORD not set: demo users NOT created in %s.", settings.environment)
            return 0
        password = DEV_DEMO_PASSWORD
    elif settings.is_protected_env and len(password) < 12:
        logger.warning("DEMO_USER_PASSWORD shorter than 12 characters: demo users NOT created in %s.", settings.environment)
        return 0
    created = 0
    for username, display, role, facility in DEMO_USERS:
        if db.query(User).filter(User.username == username).first() is None:
            db.add(User(username=username, display_name=display, role=role, facility_id=facility, password_hash=hash_password(password)))
            created += 1
    return created


def seed(reset: bool = False, today: Optional[date] = None, users: bool = False, queue: bool = False) -> Dict[str, int]:
    today = today or date.today()
    if reset:
        if settings.is_protected_env:
            raise SystemExit("Refusing --reset in staging/production.")
        reset_database()
    else:
        run_migrations()

    master = load_master()
    stats = {"patients_created": 0, "patients_updated": 0, "docs_created": 0, "docs_updated": 0, "docs_unchanged": 0}
    db = SessionLocal()
    try:
        for p in master["patients"]:
            dob = date.fromisoformat(p["date_of_birth"])
            fields = {
                "name": p["name"], "gender": p["gender"], "date_of_birth": dob,
                "age_years": age_on(dob, today), "history_facts": history_facts(p),
            }
            patient = db.query(Patient).filter(Patient.mrn == p["mrn"]).first()
            if patient is None:
                patient = Patient(mrn=p["mrn"], **fields)
                db.add(patient)
                db.flush()
                stats["patients_created"] += 1
            elif patient.is_anonymized:
                continue  # never resurrect an anonymised record
            elif any(getattr(patient, k) != v for k, v in fields.items()):
                for k, v in fields.items():
                    setattr(patient, k, v)
                stats["patients_updated"] += 1

            # Same pipeline as user uploads (validate -> store -> queue -> parse -> chunk -> embed -> index).
            # Unchanged text = same checksum = duplicate, so re-seeding is a no-op; changed text = new version.
            for doc in render_documents(p):
                try:
                    status, _job = ingest_text_inline(
                        db, patient=patient, document_id=doc.id, doc_type=doc.doc_type, title=doc.title, doc_date=doc.doc_date,
                        text=doc.content, actor="SYSTEM", source=f"{master['dataset']}@{master['version']}", is_synthetic=True,
                    )
                except ValueError:  # a clinician deleted this seeded document: respect that
                    stats["docs_skipped_deleted"] = stats.get("docs_skipped_deleted", 0) + 1
                    continue
                stats[{"created": "docs_created", "new_version": "docs_updated", "duplicate": "docs_unchanged"}[status]] += 1
        if users:
            stats["users_created"] = seed_users(db)
        db.commit()
        if stats["docs_created"] or stats["docs_updated"]:
            from backend.app.ingestion.worker import run_pending_jobs

            stats["ingestion_jobs_run"] = run_pending_jobs()  # seeded history is searchable as soon as startup ends
        if queue:
            from backend.app.services.demo_queue import seed_demo_group

            stats["queue_visits"] = seed_demo_group(db, "base")
    finally:
        db.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", action="store_true", help="Create the demo user accounts.")
    parser.add_argument("--queue", action="store_true", help="Create the demo waiting room.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables first (refused in production).")
    args = parser.parse_args()
    if args.reset:
        print("WARNING: --reset drops every table in", settings.database_url.split("@")[-1])
    print(seed(reset=args.reset, users=args.users, queue=args.queue))
