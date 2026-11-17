"""
Demo waiting room, owned by the backend (replaces the frontend's hard-coded patient lists).
Entries are idempotent: the request_id is derived from group+MRN, so re-seeding never duplicates a visit.
The LLM is skipped so seeding is fast, free and deterministic; ESI comes from ML + safety rules.
"""
import json
import os
import uuid
from datetime import timedelta
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.llm.assessor import LLMUnavailable
from backend.app.models import ACTIVE_STATUSES, Patient, Visit, utc_now
from backend.app.services.visit_service import create_visit

DEMO_QUEUE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "demo_queue.json")
_NS = uuid.UUID("6f1c1c0e-3d7a-4a3e-9d57-0c5f6b2a9e11")


def _skip_llm(_inp):
    raise LLMUnavailable("llm_skipped_demo_seed")


def load_demo_queue() -> dict:
    with open(DEMO_QUEUE_PATH, encoding="utf-8") as f:
        return json.load(f)


def seed_demo_group(db: Session, group: str, actor: str = "SYSTEM", facility_id: Optional[str] = None) -> int:
    """Creates (or re-activates) the visits of one group ('base' or 'surge'). Returns how many changed."""
    facility_id = facility_id or settings.facility_id
    changed = 0
    for entry in load_demo_queue()[group]:
        patient = db.query(Patient).filter(Patient.mrn == entry["mrn"]).first()
        if patient is None or patient.is_anonymized:
            continue
        request_id = str(uuid.uuid5(_NS, f"{group}:{entry['mrn']}"))
        arrival = utc_now() - timedelta(minutes=entry["arrival_minutes_ago"])
        existing = db.query(Visit).filter(Visit.request_id == request_id).first()
        if existing:
            if existing.status == "SURGE_ENDED":
                existing.status, existing.arrival_time, existing.last_assessed_at = "WAITING", arrival, utc_now()
                changed += 1
            continue
        create_visit(
            db, patient=patient, is_new=False, complaint=entry["complaint"], vitals=entry["vitals"], request_id=request_id,
            req_hash="demo", facility_id=facility_id, actor=actor, llm_fn=_skip_llm, source="SEED", arrival_time=arrival,
            is_demo_surge=group == "surge",
        )
        changed += 1
    db.commit()
    return changed


def end_surge(db: Session, facility_id: str, actor: str) -> int:
    visits = db.query(Visit).filter(Visit.is_demo_surge.is_(True), Visit.facility_id == facility_id, Visit.status.in_(ACTIVE_STATUSES)).all()
    for v in visits:
        v.status = "SURGE_ENDED"
    db.commit()
    return len(visits)
