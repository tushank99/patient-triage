import uuid
import time
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import Base, engine, get_db
from backend.app.models import Patient, Visit, AuditLog
from backend.app.schemas import PatientIntake, OverrideRequest, TriageResult
from backend.app.services.assessment_engine import execute_triage_assessment
from backend.app.background_tasks import deterioration_monitor

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    Base.metadata.create_all(bind=engine)
    monitor_task = asyncio.create_task(deterioration_monitor())
    yield
    
    monitor_task.cancel()

app = FastAPI(
    title="PatientTriage.ai",
    description="Enterprise AI Patient Triage Platform Prototype",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173", "http://127.0.0.1:8080", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Idempotency Cache (Prevents double clicks)
# Maps MRN to a timestamp to prevent duplicate submissions within 10 seconds.
recent_submissions = {}


@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "PatientTriage.ai Backend is running."}

@app.post("/api/triage", response_model=TriageResult)
async def process_triage(intake: PatientIntake, db: Session = Depends(get_db)):
    """
    Main Async Triage Engine.
    """
    mrn = intake.mrn or "UNKNOWN"
    current_time = time.time()
    if mrn in recent_submissions and (current_time - recent_submissions[mrn]) < 10:
        raise HTTPException(status_code=409, detail="Duplicate submission detected. Please wait.")
    recent_submissions[mrn] = current_time

    patient_data = {
        "mrn": intake.mrn,
        "age": intake.age_years,
        "hr": intake.vitals.heart_rate,
        "bpSys": intake.vitals.systolic_bp,
        "bpDia": intake.vitals.diastolic_bp,
        "spo2": intake.vitals.oxygen_saturation,
        "temp": intake.vitals.temperature_celsius,
        "pain": intake.vitals.pain_score,
    }
    
    # Run the real ML + LLM pipeline
    ai_result = execute_triage_assessment(
        patient=patient_data,
        scrubbed_complaint=intake.chief_complaint, 
        ehr_history=intake.prior_history_summary or "No prior history available.",
        routing={"model_route": "ADULT_MODEL" if intake.age_years >= 18 else "PEDIATRIC_MODEL", "uncertainty_penalty": 0.0}
    )
    
    # 2. Database Insertion (Atomic transaction)
    # Check if patient exists, else create
    patient = db.query(Patient).filter(Patient.mrn == intake.mrn).first() if intake.mrn else None
    if not patient:
        patient = Patient(
            mrn=intake.mrn or f"WALK-IN-{uuid.uuid4().hex[:6].upper()}",
            age_years=intake.age_years,
            gender=intake.gender,
            has_prior_history=intake.has_prior_history,
            prior_history_summary=intake.prior_history_summary
        )
        db.add(patient)
        db.flush()

    # Create the Visit record
    visit = Visit(
        patient_id=patient.id,
        chief_complaint=intake.chief_complaint,
        heart_rate=intake.vitals.heart_rate,
        systolic_bp=intake.vitals.systolic_bp,
        diastolic_bp=intake.vitals.diastolic_bp,
        respiratory_rate=intake.vitals.respiratory_rate,
        oxygen_saturation=intake.vitals.oxygen_saturation,
        temperature_celsius=intake.vitals.temperature_celsius,
        pain_score=intake.vitals.pain_score,
        
        ai_esi_score=ai_result["recommended_esi"],
        ai_confidence=ai_result["confidence_score"] / 100.0, # DB stores as 0.0 - 1.0
        ai_rationale=ai_result["rationale"],
        ai_risk_factors=[],
        
        final_esi_score=ai_result["recommended_esi"], # Defaults to AI suggestion until nurse overrides
        status="WAITING"
    )
    db.add(visit)
    db.commit()
    db.refresh(visit)

    return TriageResult(
        visit_id=visit.id, # type: ignore
        recommended_esi=ai_result["recommended_esi"],
        confidence_score=ai_result["confidence_score"],
        rationale=ai_result["rationale"],
        badges=ai_result.get("badges", [])
    )

@app.post("/api/override")
def nurse_override(override: OverrideRequest, db: Session = Depends(get_db)):
    """
    Immutable Audit Endpoint. The nurse MUST provide a rationale.
    """
    if not override.override_reason.strip():
        raise HTTPException(status_code=400, detail="A clinical override reason is strictly required.")

    visit = db.query(Visit).filter(Visit.id == override.visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found.")

    # Create Audit Log (Snapshotting the exact AI ESI at this exact moment)
    log = AuditLog(
        visit_id=visit.id,
        clinician_id=override.clinician_id,
        action="OVERRIDE_ESI",
        initial_ai_esi=visit.ai_esi_score,
        assigned_esi=override.assigned_esi,
        override_reason=override.override_reason
    )
    
    # Update Visit
    visit.final_esi_score = override.assigned_esi # type: ignore
    visit.is_overridden = True # type: ignore
    
    db.add(log)
    db.commit()
    
    return {"status": "success", "message": "Override legally logged and applied."}

@app.get("/api/queue")
def get_waiting_room(db: Session = Depends(get_db)):
    """
    Fetches the live waiting room, sorted by Acuity (ESI 1 is highest priority), then arrival time.
    """
    visits = db.query(Visit).join(Patient).filter(
        Visit.status.in_(["WAITING", "NEEDS_REASSESSMENT"])
    ).order_by(Visit.final_esi_score.asc(), Visit.arrival_time.asc()).all()
    
    results = []
    for v in visits:
        results.append({
            "visit_id": v.id,
            "patient_mrn": v.patient.mrn,
            "age": v.patient.age_years,
            "complaint": v.chief_complaint,
            "esi_score": v.final_esi_score,
            "status": v.status,
            "wait_time_minutes": round((datetime.now(timezone.utc) - v.arrival_time).total_seconds() / 60, 1),
            "is_overridden": v.is_overridden
        })
    return results

@app.post("/api/simulate-surge")
def simulate_surge(background_tasks: BackgroundTasks):
    """
    The Wow Factor: Instantly floods the queue by running the seeder script in the background.
    """
    import subprocess
    # Run the seed script in the background so the API doesn't hang
    background_tasks.add_task(subprocess.run, ["python", "-m", "backend.scripts.seed_data"])
    return {"status": "Surge Initiated", "message": "15+ patients are being injected into the queue."}


@app.post("/api/archive")
def archive_patient(visit_id: str, disposition: str, db: Session = Depends(get_db)):
    """
    Phase 5: Moves a patient from the active queue to the historical archive.
    """
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found.")

    if visit.status != "WAITING":
        raise HTTPException(status_code=400, detail="Patient is already archived.")

    # Update visit status to the final disposition
    visit.status = disposition  # e.g., "Discharged", "Admitted", "LWBS"
    visit.discharge_time = datetime.now(timezone.utc)
    
    # Log the disposition in the audit trail
    log = AuditLog(
        visit_id=visit.id,
        clinician_id="SYSTEM",
        action="PATIENT_DISPOSITIONED",
        initial_ai_esi=visit.ai_esi_score,
        assigned_esi=visit.final_esi_score,
        override_reason=f"Patient removed from queue. Disposition: {disposition}"
    )
    db.add(log)
    db.commit()
    
    return {"status": "success", "message": f"Patient archived as {disposition}."}


@app.get("/api/archive")
def get_archived_records(db: Session = Depends(get_db)):
    """
    Phase 5: Fetches the historical patient archive.
    Enforces a strict 7-day boundary to prevent database freezing on large queries.
    """
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    
    # Query only visits that are NOT waiting and were discharged in the last 7 days
    archived_visits = db.query(Visit).join(Patient).filter(
        Visit.status != "WAITING",
        Visit.discharge_time >= seven_days_ago
    ).order_by(Visit.discharge_time.desc()).all()
    
    results = []
    for v in archived_visits:
        results.append({
            "visit_id": v.id,
            "patient_mrn": v.patient.mrn,
            "patient_name": "REDACTED" if getattr(v.patient, 'is_anonymized', False) else getattr(v.patient, 'name', 'Unknown'),
            "age": v.patient.age_years,
            "complaint": v.chief_complaint,
            "esi_score": v.final_esi_score,
            "disposition": v.status,
            "disposition_time": v.discharge_time,
            "is_overridden": v.is_overridden,
            "is_anonymized": getattr(v.patient, 'is_anonymized', False)
        })
        
    return results


@app.post("/api/archive/revert")
def revert_archived_patient(visit_id: str, db: Session = Depends(get_db)):
    """
    Phase 5: The 15-Minute Undo Function.
    Reverts an accidentally discharged patient back to the active queue.
    """
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found.")

    if visit.status == "WAITING":
        raise HTTPException(status_code=400, detail="Patient is already in the active queue.")
        
    if not visit.discharge_time:
         raise HTTPException(status_code=400, detail="Cannot revert: No discharge time recorded.")

    # Enforce the strict 15-minute compliance window
    minutes_since_discharge = (datetime.now(timezone.utc) - visit.discharge_time).total_seconds() / 60
    if minutes_since_discharge > 15:
        raise HTTPException(status_code=403, detail="Compliance Error: The 15-minute revert window has expired.")

    # Revert back to waiting room
    old_disposition = visit.status
    visit.status = "WAITING"
    visit.discharge_time = None
    
    # Log the undo action immutably
    log = AuditLog(
        visit_id=visit.id,
        clinician_id="SYSTEM",
        action="DISPOSITION_REVERTED",
        initial_ai_esi=visit.ai_esi_score,
        assigned_esi=visit.final_esi_score,
        override_reason=f"Accidental disposition ({old_disposition}) reverted by clinician within 15m window."
    )
    db.add(log)
    db.commit()
    
    return {"status": "success", "message": "Patient successfully returned to the active queue."}


@app.post("/api/archive/anonymize")
def anonymize_patient_record(mrn: str, db: Session = Depends(get_db)):
    """
    Phase 5: GDPR / HIPAA Soft-Delete Protocol.
    Purges PII (Name, MRN) but strictly retains clinical vitals and ESI scores for ML retraining.
    """
    patient = db.query(Patient).filter(Patient.mrn == mrn).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    if getattr(patient, 'is_anonymized', False):
        return {"status": "success", "message": "Patient is already anonymized."}

    # Soft-delete PII
    patient.mrn = f"REDACTED_{uuid.uuid4().hex[:8]}"
    if hasattr(patient, 'name'):
        patient.name = "REDACTED"
    
    # Add flag if it exists on model, otherwise we rely on the REDACTED string
    if hasattr(patient, 'is_anonymized'):
        patient.is_anonymized = True
        
    db.commit()
    
    return {"status": "success", "message": "PII successfully purged. Clinical data retained for ML."}
