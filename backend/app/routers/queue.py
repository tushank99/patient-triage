import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, contains_eager

from backend.app.auth import PERMISSIONS, authorize_visit, get_current_user, log_access, require
from backend.app.database import get_db
from backend.app.ingestion.service import purge_patient_documents
from backend.app.models import ACTIVE_STATUSES, NON_ARCHIVE_STATUSES, AuditLog, Patient, User, Visit, as_utc, utc_now
from backend.app.schemas import AnonymizeRequest, ArchiveItem, DispositionRequest, QueueItem, RevertRequest, StatusResponse

router = APIRouter(tags=["queue"])


def _may_see_names(user: User) -> bool:
    return user.role in PERMISSIONS["patient:read"]


@router.get("/queue", response_model=List[QueueItem])
def get_waiting_room(user: User = Depends(require("queue:read")), db: Session = Depends(get_db)):
    """Live waiting room for the user's facility, most acute first, then longest waiting."""
    visits = (
        db.query(Visit).join(Patient).options(contains_eager(Visit.patient))  # fill v.patient from this JOIN: no per-row lazy load
        .filter(Visit.status.in_(ACTIVE_STATUSES), Visit.facility_id == user.facility_id)
        .order_by(Visit.final_esi_score.asc(), Visit.arrival_time.asc()).all()
    )
    now, names = datetime.now(timezone.utc), _may_see_names(user)
    return [
        QueueItem(
            visit_id=v.id, patient_mrn=v.patient.mrn, patient_name=v.patient.name if names else None,
            age=v.patient.age_years, gender=v.patient.gender, complaint=v.chief_complaint,  # type: ignore[arg-type]
            esi_score=v.final_esi_score, ai_esi_score=v.ai_esi_score, ai_model_probability=v.ai_confidence, status=v.status,
            wait_time_minutes=round((now - as_utc(v.arrival_time)).total_seconds() / 60, 1), arrival_time=v.arrival_time,
            is_overridden=v.is_overridden, needs_review=v.needs_review, review_reasons=v.review_reasons or [],
            is_demo_surge=v.is_demo_surge,
        )
        for v in visits
    ]


@router.post("/archive", response_model=StatusResponse)
def archive_patient(req: DispositionRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db)):
    """Moves a visit from the active queue (WAITING or NEEDS_REASSESSMENT) to the archive."""
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == req.visit_id).first(), "VISIT_DISPOSITION")
    if visit.status not in ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail="Patient is already archived.")
    visit.status, visit.discharge_time = req.disposition, utc_now()
    db.add(AuditLog(
        visit_id=visit.id, clinician_id=user.username, action="PATIENT_DISPOSITIONED", initial_ai_esi=visit.ai_esi_score,
        assigned_esi=visit.final_esi_score or visit.ai_esi_score or 3,
        override_reason=f"Patient removed from queue. Disposition: {req.disposition}",
        raw_payload_snapshot={"disposition": req.disposition, "checklist": req.checklist},
    ))
    db.commit()
    return StatusResponse(status="success", message=f"Patient archived as {req.disposition}.")


@router.get("/archive", response_model=List[ArchiveItem])
def get_archived_records(user: User = Depends(require("archive:read")), db: Session = Depends(get_db)):
    """Historical archive, limited to the last 7 days. Roles without patient-level access see no names."""
    since = datetime.now(timezone.utc) - timedelta(days=7)
    visits = (
        db.query(Visit).join(Patient).options(contains_eager(Visit.patient))
        .filter(Visit.status.notin_(ACTIVE_STATUSES + NON_ARCHIVE_STATUSES), Visit.discharge_time >= since, Visit.facility_id == user.facility_id)
        .order_by(Visit.discharge_time.desc()).all()
    )
    names = _may_see_names(user)
    return [
        ArchiveItem(
            visit_id=v.id, patient_mrn=v.patient.mrn,
            patient_name=None if not names else ("REDACTED" if v.patient.is_anonymized else v.patient.name),
            age=v.patient.age_years, gender=v.patient.gender, complaint=v.chief_complaint, esi_score=v.final_esi_score,  # type: ignore[arg-type]
            ai_esi_score=v.ai_esi_score, ai_model_probability=v.ai_confidence, disposition=v.status, arrival_time=v.arrival_time,
            disposition_time=v.discharge_time, is_overridden=v.is_overridden, is_anonymized=v.patient.is_anonymized,
        )
        for v in visits
    ]


@router.post("/archive/revert", response_model=StatusResponse)
def revert_archived_patient(req: RevertRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db)):
    """15-minute undo for an accidental disposition."""
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == req.visit_id).first(), "VISIT_REVERT")
    if visit.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail="Patient is already in the active queue.")
    if not visit.discharge_time:
        raise HTTPException(status_code=400, detail="Cannot revert: no discharge time recorded.")
    if (datetime.now(timezone.utc) - as_utc(visit.discharge_time)).total_seconds() / 60 > 15:
        raise HTTPException(status_code=403, detail="Compliance Error: the 15-minute revert window has expired.")
    old = visit.status
    visit.status, visit.discharge_time = "WAITING", None
    visit.last_assessed_at = utc_now()
    db.add(AuditLog(
        visit_id=visit.id, clinician_id=user.username, action="DISPOSITION_REVERTED", initial_ai_esi=visit.ai_esi_score,
        assigned_esi=visit.final_esi_score or visit.ai_esi_score or 3,
        override_reason=f"Accidental disposition ({old}) reverted within 15m window.",
    ))
    db.commit()
    return StatusResponse(status="success", message="Patient returned to the active queue.")


@router.post("/archive/anonymize", response_model=StatusResponse)
def anonymize_patient_record(req: AnonymizeRequest, user: User = Depends(require("patient:anonymize")), db: Session = Depends(get_db)):
    """Soft delete (admin only): purges name/MRN and history documents; keeps visit vitals and ESI."""
    patient = db.query(Patient).filter(Patient.mrn == req.mrn).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    if patient.is_anonymized:
        return StatusResponse(status="success", message="Patient is already anonymized.")
    log_access(db, user, "PATIENT_ANONYMIZE", patient=patient)
    purge_patient_documents(db, patient)  # documents, versions, chunks, vectors and stored originals
    patient.mrn, patient.name = f"REDACTED_{uuid.uuid4().hex[:8]}", "REDACTED"
    patient.date_of_birth, patient.history_facts, patient.is_anonymized = None, None, True
    db.commit()
    return StatusResponse(status="success", message="PII and history documents purged. Clinical visit data retained.")
