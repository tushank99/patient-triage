"""Visit lifecycle helpers shared by the routers and the demo-queue seeder."""
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.models import AuditLog, Patient, Visit, VitalsReading, utc_now
from backend.app.schemas import TriageRequest, TriageResponse
from backend.app.services.history_store import retrieve_history
from backend.app.services.triage_pipeline import Assessment, AssessmentInput, LLMFn, run_assessment

VITAL_KEYS = ("heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "oxygen_saturation", "temperature_celsius", "pain_score")


def request_hash(req: TriageRequest) -> str:
    body = req.model_dump(mode="json", exclude={"request_id"})
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def visit_vitals(visit: Visit) -> Dict[str, Any]:
    return {k: getattr(visit, k) for k in VITAL_KEYS}


def _apply_vitals(visit: Visit, vitals: dict) -> None:
    for k in VITAL_KEYS:
        setattr(visit, k, vitals.get(k))


def _store_ai(visit: Visit, a: Assessment) -> None:
    visit.ai_esi_score = a.recommended_esi
    visit.ai_confidence = a.confidence.model_probability
    visit.ai_rationale = a.rationale
    visit.ai_risk_factors = a.risk_factors
    visit.needs_review = a.requires_review
    visit.review_reasons = a.review_reasons
    visit.reviewed_by = None
    visit.reviewed_at = None
    visit.last_assessed_at = utc_now()


def build_response(
    visit: Visit, patient: Patient, a: Assessment, *, is_new: bool, requires_confirmation: bool = False,
    warnings: Optional[List[str]] = None,
) -> TriageResponse:
    return TriageResponse(
        visit_id=visit.id, mrn=patient.mrn, is_new_patient=is_new, recommended_esi=a.recommended_esi,
        final_esi=visit.final_esi_score or a.recommended_esi, requires_confirmation=requires_confirmation,
        requires_clinician_review=a.requires_review, review_reasons=a.review_reasons, confidence=a.confidence,
        rationale=a.rationale, risk_factors=a.risk_factors, badges=a.badges, components=a.components,
        decision_trace=a.decision_trace, history=a.history, degraded=a.degraded, warnings=warnings or [],
        pii_redactions=a.pii_redactions, versions=a.versions, claims=a.claims, grounding=a.grounding or {},
        facts=a.facts or {}, security_flags=a.security_flags,
    )


def _audit_snapshot(a: Assessment, response: TriageResponse, request_id: str, inputs: dict, extra: Optional[dict] = None) -> dict:
    """Everything needed to reconstruct and defend this assessment later."""
    snap: Dict[str, Any] = {
        "request_id": request_id,
        "inputs": inputs,
        "assessment": response.model_dump(mode="json"),
        "llm": None,
    }
    if a.llm_meta:
        meta = asdict(a.llm_meta)
        meta["request_text"] = a.llm_request_text  # the exact de-identified text that was sent out
        snap["llm"] = meta
    if extra:
        snap.update(extra)
    return snap


def record_vitals(db: Session, visit: Visit, submitted: dict, actor: str, source: str, esi_after: Optional[int]) -> None:
    db.add(VitalsReading(visit_id=visit.id, recorded_by=actor, source=source, ai_esi_after=esi_after, **{k: submitted.get(k) for k in VITAL_KEYS}))


def create_visit(
    db: Session, *, patient: Patient, is_new: bool, complaint: str, vitals: dict, request_id: str, req_hash: str,
    facility_id: str, actor: str, llm_fn: LLMFn, source: str = "INTAKE", patient_name: Optional[str] = None,
    arrival_time: Optional[datetime] = None, is_demo_surge: bool = False, warnings: Optional[List[str]] = None,
) -> TriageResponse:
    from backend.app.ops.observability import stage

    with stage("retrieval"):
        history = retrieve_history(db, patient, complaint)
    a = run_assessment(
        AssessmentInput(patient.age_years, patient.gender, complaint, vitals, patient_name or patient.name, registered=not is_new),
        history, llm_fn,
    )
    visit = Visit(
        patient_id=patient.id, facility_id=facility_id, chief_complaint=complaint, request_id=request_id,
        request_hash=req_hash, final_esi_score=a.recommended_esi, status="WAITING", is_demo_surge=is_demo_surge,
    )
    if arrival_time:
        visit.arrival_time = arrival_time
    _apply_vitals(visit, vitals)
    _store_ai(visit, a)
    db.add(visit)
    db.flush()

    response = build_response(visit, patient, a, is_new=is_new, warnings=warnings)
    visit.ai_assessment = response.model_dump(mode="json")
    record_vitals(db, visit, vitals, actor, source, a.recommended_esi)
    db.add(AuditLog(
        visit_id=visit.id, clinician_id=actor, action="AI_TRIAGE", initial_ai_esi=a.recommended_esi,
        assigned_esi=a.recommended_esi,
        raw_payload_snapshot=_audit_snapshot(
            a, response, request_id,
            {"age_years": patient.age_years, "gender": patient.gender, "complaint": complaint, "vitals": vitals, "registered": not is_new},
        ),
    ))
    return response


def reassess_visit(
    db: Session, visit: Visit, *, new_vitals: dict, complaint: Optional[str], request_id: str, actor: str, llm_fn: LLMFn
) -> TriageResponse:
    patient = visit.patient
    if complaint:
        visit.chief_complaint = complaint
    measured = {k: v for k, v in new_vitals.items() if v is not None}
    merged = {**visit_vitals(visit), **measured}  # keep earlier readings for anything not re-measured
    from backend.app.ops.observability import stage

    with stage("retrieval"):
        history = retrieve_history(db, patient, visit.chief_complaint)
    a = run_assessment(
        AssessmentInput(patient.age_years, patient.gender, visit.chief_complaint, merged, patient.name, registered=True),
        history, llm_fn,
    )
    previous_final = visit.final_esi_score or a.recommended_esi
    requires_confirmation = a.recommended_esi > previous_final  # a downgrade is never automatic
    if not requires_confirmation:
        visit.final_esi_score = a.recommended_esi
    _apply_vitals(visit, merged)
    _store_ai(visit, a)
    visit.status = "WAITING"
    visit.last_reassess_request_id = request_id

    response = build_response(visit, patient, a, is_new=False, requires_confirmation=requires_confirmation)
    visit.ai_assessment = response.model_dump(mode="json")
    record_vitals(db, visit, measured, actor, "REASSESS", a.recommended_esi)
    db.add(AuditLog(
        visit_id=visit.id, clinician_id=actor, action="AI_REASSESS", initial_ai_esi=a.recommended_esi,
        assigned_esi=visit.final_esi_score or a.recommended_esi,
        raw_payload_snapshot=_audit_snapshot(
            a, response, request_id,
            {"new_vitals": measured, "merged_vitals": merged, "complaint": visit.chief_complaint},
            {"previous_final_esi": previous_final, "downgrade_pending_confirmation": requires_confirmation},
        ),
    ))
    return response
