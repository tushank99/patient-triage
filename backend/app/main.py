import uuid
import time
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import Base, engine, get_db
from backend.app.models import Patient, Visit, AuditLog
from backend.app.schemas import PatientIntake, OverrideRequest, TriageResult
from backend.app.services.fusion_engine import generate_triage_recommendation
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

    ai_result = await generate_triage_recommendation(
        age=intake.age_years,
        vitals=intake.vitals.model_dump(),
        complaint=intake.chief_complaint,
        has_history=intake.has_prior_history
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
        
        ai_esi_score=ai_result["suggested_esi"],
        ai_confidence=ai_result["confidence"],
        ai_rationale=ai_result["rationale"],
        ai_risk_factors=ai_result["key_drivers"],
        
        final_esi_score=ai_result["suggested_esi"], # Defaults to AI suggestion until nurse overrides
        status="WAITING"
    )
    db.add(visit)
    db.commit()
    db.refresh(visit)

    return TriageResult(
        visit_id=visit.id, # type: ignore
        suggested_esi=ai_result["suggested_esi"],
        confidence=ai_result["confidence"],
        rationale=ai_result["rationale"],
        key_drivers=ai_result["key_drivers"],
        is_escalated_failsafe=ai_result["is_escalated_failsafe"]
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
