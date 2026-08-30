import uuid 
from datetime import datetime, timezone 
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey 
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database import Base 

def utc_now(): 
    """Helper to return current timezone-aware UTC time.""" 
    return datetime.now(timezone.utc)

class Patient(Base): 
    __tablename__ = "patients" 
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mrn: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True) # Added for Phase 5 Archive
    age_years: Mapped[float] = mapped_column(Float, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    has_prior_history: Mapped[bool] = mapped_column(Boolean, default=False)
    prior_history_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_anonymized: Mapped[bool] = mapped_column(Boolean, default=False) # Phase 5 GDPR Flag
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    
    visits = relationship("Visit", back_populates="patient")

class Visit(Base): 
    __tablename__ = "visits" 
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    arrival_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    
    chief_complaint: Mapped[str] = mapped_column(Text, nullable=False)
    heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    systolic_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diastolic_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    respiratory_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    oxygen_saturation: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_celsius: Mapped[float | None] = mapped_column(Float, nullable=True)
    pain_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    ai_esi_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_risk_factors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    final_esi_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="WAITING")
    is_overridden: Mapped[bool] = mapped_column(Boolean, default=False)
    discharge_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True) # Phase 5 15-min limit
    
    patient = relationship("Patient", back_populates="visits")
    audit_logs = relationship("AuditLog", back_populates="visit")

class AuditLog(Base): 
    __tablename__ = "audit_logs" 
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    visit_id: Mapped[str] = mapped_column(String, ForeignKey("visits.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    clinician_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    
    initial_ai_esi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assigned_esi: Mapped[int] = mapped_column(Integer, nullable=False)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_payload_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    visit = relationship("Visit", back_populates="audit_logs")