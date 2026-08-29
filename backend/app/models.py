import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

# Use relative import to resolve Pylance warning
from backend.app.database import Base
def utc_now():
    """Helper to return current timezone-aware UTC time."""
    return datetime.now(timezone.utc)

class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mrn = Column(String, unique=True, index=True, nullable=False) # Medical Record Number
    age_years = Column(Float, nullable=False)
    gender = Column(String, nullable=False)
    has_prior_history = Column(Boolean, default=False)
    prior_history_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    visits = relationship("Visit", back_populates="patient")

class Visit(Base):
    __tablename__ = "visits"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    arrival_time = Column(DateTime(timezone=True), default=utc_now)
    
    # Raw & Extracted Clinical Inputs
    chief_complaint = Column(Text, nullable=False)
    heart_rate = Column(Integer, nullable=True)
    systolic_bp = Column(Integer, nullable=True)
    diastolic_bp = Column(Integer, nullable=True)
    respiratory_rate = Column(Integer, nullable=True)
    oxygen_saturation = Column(Float, nullable=True)
    temperature_celsius = Column(Float, nullable=True)
    pain_score = Column(Integer, nullable=True)
    
    # AI Output Fields
    ai_esi_score = Column(Integer, nullable=True)       # 1 to 5 (ESI Scale)
    ai_confidence = Column(Float, nullable=True)        # 0.0 to 1.0
    ai_rationale = Column(Text, nullable=True)
    ai_risk_factors = Column(JSON, nullable=True)      # SHAP / Key features
    
    # Final Decision & Queue Status
    final_esi_score = Column(Integer, nullable=True)
    status = Column(String, default="WAITING")          # WAITING, IN_TRIAGE, ESCALATED, COMPLETED
    is_overridden = Column(Boolean, default=False)
    
    patient = relationship("Patient", back_populates="visits")
    audit_logs = relationship("AuditLog", back_populates="visit")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    visit_id = Column(String, ForeignKey("visits.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utc_now)
    clinician_id = Column(String, nullable=False)
    action = Column(String, nullable=False)             # e.g., "OVERRIDE_ESI", "CONFIRM_ESI"
    initial_ai_esi = Column(Integer, nullable=True)
    assigned_esi = Column(Integer, nullable=False)
    override_reason = Column(Text, nullable=True)       # Clinical justification
    raw_payload_snapshot = Column(JSON, nullable=True)

    visit = relationship("Visit", back_populates="audit_logs")
