import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth import PERMISSIONS, authorize_patient, authorize_visit, log_access, require
from backend.app.database import get_db
from backend.app.deps import get_llm_fn
from backend.app.models import ACTIVE_STATUSES, AuditLog, Patient, User, Visit, utc_now
from backend.app.schemas import (
    AuditEventOut,
    OverrideRequest,
    OverrideResponse,
    PatientLookup,
    ReassessRequest,
    ReviewRequest,
    StatusResponse,
    TriageRequest,
    TriageResponse,
    VitalsReadingOut,
)
from backend.app.services.history_store import count_documents
from backend.app.services.triage_pipeline import LLMFn
from backend.app.services.visit_service import create_visit, reassess_visit, request_hash

router = APIRouter(tags=["triage"])


@router.get("/patients/{mrn}", response_model=PatientLookup)
def lookup_patient(mrn: str, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    patient = authorize_patient(db, user, db.query(Patient).filter(Patient.mrn == mrn).first(), "PATIENT_LOOKUP")
    n = count_documents(db, patient)
    db.commit()
    return PatientLookup(
        mrn=patient.mrn, name=patient.name, gender=patient.gender, date_of_birth=patient.date_of_birth,  # type: ignore[arg-type]
        age_years=patient.age_years, has_history=n > 0, document_count=n,
    )


@router.post("/triage", response_model=TriageResponse)
def process_triage(
    req: TriageRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db), llm_fn: LLMFn = Depends(get_llm_fn)
):
    request_id, req_hash = str(req.request_id), request_hash(req)

    existing = db.query(Visit).filter(Visit.request_id == request_id).first()
    if existing:
        if existing.request_hash != req_hash or existing.facility_id != user.facility_id:
            raise HTTPException(status_code=409, detail="request_id was already used with a different payload.")
        return TriageResponse.model_validate(existing.ai_assessment)

    if req.transcript_capture_id:
        from backend.app.intake.captures import CaptureError, check_transcript_for_triage

        try:
            check_transcript_for_triage(db, req.transcript_capture_id)
        except CaptureError as e:
            raise HTTPException(status_code=e.status, detail=e.message)

    warnings: List[str] = []
    is_new = False
    if req.mrn:
        patient = authorize_patient(db, user, db.query(Patient).filter(Patient.mrn == req.mrn).first(), "PATIENT_TRIAGE")
        if abs(patient.age_years - req.age_years) > 1.5 or patient.gender != req.gender:
            warnings.append("Age or sex differs from the record on file for this MRN. Confirm patient identity.")
    else:
        is_new = True
        patient = Patient(mrn=f"WALK-{uuid.uuid4().hex[:8].upper()}", name=req.patient_name, age_years=req.age_years, gender=req.gender)
        db.add(patient)
        db.flush()

    try:  # a concurrent retry with the same request_id can lose the race at flush or at commit
        response = create_visit(
            db, patient=patient, is_new=is_new, complaint=req.chief_complaint, vitals=req.vitals.model_dump(), request_id=request_id,
            req_hash=req_hash, facility_id=user.facility_id, actor=user.username, llm_fn=llm_fn, patient_name=req.patient_name,
            warnings=warnings,
        )
        db.commit()
    except IntegrityError:
        db.rollback()
        winner = db.query(Visit).filter(Visit.request_id == request_id).first()
        if winner and winner.request_hash == req_hash:
            return TriageResponse.model_validate(winner.ai_assessment)
        raise HTTPException(status_code=409, detail="Conflicting concurrent submission.")
    return response


@router.post("/visits/{visit_id}/reassess", response_model=TriageResponse)
def reassess(
    visit_id: str, req: ReassessRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db),
    llm_fn: LLMFn = Depends(get_llm_fn),
):
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == visit_id).first(), "VISIT_REASSESS")
    if visit.status not in ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail="Visit is no longer in the active queue.")
    if visit.last_reassess_request_id == str(req.request_id) and visit.ai_assessment:
        return TriageResponse.model_validate(visit.ai_assessment)
    response = reassess_visit(
        db, visit, new_vitals=req.vitals.model_dump(), complaint=req.chief_complaint, request_id=str(req.request_id),
        actor=user.username, llm_fn=llm_fn,
    )
    db.commit()
    return response


@router.get("/visits/{visit_id}/vitals", response_model=List[VitalsReadingOut])
def vitals_history(visit_id: str, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == visit_id).first(), "VISIT_VITALS_READ")
    db.commit()
    return [VitalsReadingOut.model_validate(r, from_attributes=True) for r in visit.vitals_readings]


@router.get("/visits/{visit_id}/audit", response_model=List[AuditEventOut])
def audit_trail(visit_id: str, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == visit_id).first(), "VISIT_AUDIT_READ")
    events = db.query(AuditLog).filter(AuditLog.visit_id == visit.id).order_by(AuditLog.timestamp.asc()).all()
    db.commit()
    return [
        AuditEventOut(
            timestamp=e.timestamp, actor=e.clinician_id, action=e.action, initial_ai_esi=e.initial_ai_esi,
            assigned_esi=e.assigned_esi, reason=e.override_reason, detail=e.raw_payload_snapshot,
        )
        for e in events
    ]


@router.post("/visits/{visit_id}/review", response_model=StatusResponse)
def acknowledge_review(
    visit_id: str, req: ReviewRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db)
):
    """A clinician confirms they have looked at an assessment the system flagged for review."""
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == visit_id).first(), "VISIT_REVIEW")
    if not visit.needs_review:
        return StatusResponse(status="success", message="Nothing to review.")
    visit.needs_review, visit.reviewed_by, visit.reviewed_at = False, user.username, utc_now()
    db.add(AuditLog(
        visit_id=visit.id, clinician_id=user.username, action="REVIEW_ACKNOWLEDGED", initial_ai_esi=visit.ai_esi_score,
        assigned_esi=visit.final_esi_score or visit.ai_esi_score or 3, override_reason=req.note,
        raw_payload_snapshot={"review_reasons": visit.review_reasons},
    ))
    db.commit()
    return StatusResponse(status="success", message="Review acknowledged.")


@router.post("/override", response_model=OverrideResponse)
def nurse_override(override: OverrideRequest, user: User = Depends(require("override:write")), db: Session = Depends(get_db)):
    """Human override. Anyone clinical may raise acuity or lower it down to the safety-rule floor; going
    below the floor needs a charge nurse or physician. Always audited with the AI's state at that moment."""
    if not override.override_reason.strip():
        raise HTTPException(status_code=400, detail="A clinical override reason is strictly required.")
    visit = authorize_visit(db, user, db.query(Visit).filter(Visit.id == override.visit_id).first(), "VISIT_OVERRIDE")
    if visit.status not in ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail="Visit is no longer in the active queue.")

    floor = ((visit.ai_assessment or {}).get("components") or {}).get("rule_floor_esi")
    below_floor = floor is not None and override.assigned_esi > floor
    if below_floor and user.role not in PERMISSIONS["override:below_floor"]:
        log_access(db, user, "DENIED:override:below_floor", patient=visit.patient, visit_id=visit.id, allowed=False)
        db.commit()
        raise HTTPException(status_code=403, detail=f"Safety rules require ESI {floor} or more acute. Lowering it needs a charge nurse or physician.")

    previous = visit.final_esi_score
    db.add(AuditLog(
        visit_id=visit.id, clinician_id=user.username, action="OVERRIDE_ESI_BELOW_RULE_FLOOR" if below_floor else "OVERRIDE_ESI",
        initial_ai_esi=visit.ai_esi_score, assigned_esi=override.assigned_esi, override_reason=override.override_reason,
        raw_payload_snapshot={
            "previous_final_esi": previous, "rule_floor_esi": floor, "below_rule_floor": below_floor, "role": user.role,
            "ai_assessment_at_override": visit.ai_assessment,
        },
    ))
    visit.final_esi_score, visit.is_overridden = override.assigned_esi, True
    db.commit()
    return OverrideResponse(status="success", message="Override logged and applied.", final_esi=override.assigned_esi, below_rule_floor=below_floor)
