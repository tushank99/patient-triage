from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class VitalSigns(BaseModel):
    heart_rate: Optional[int] = Field(None, ge=20, le=300)
    systolic_bp: Optional[int] = Field(None, ge=40, le=300)
    diastolic_bp: Optional[int] = Field(None, ge=20, le=200)
    respiratory_rate: Optional[int] = Field(None, ge=4, le=80)
    oxygen_saturation: Optional[float] = Field(None, ge=40.0, le=100.0)
    temperature_celsius: Optional[float] = Field(None, ge=30.0, le=45.0)
    pain_score: Optional[int] = Field(None, ge=0, le=10)

class PatientIntake(BaseModel):
    mrn: Optional[str] = None
    age_years: float = Field(..., ge=0.0, le=125.0)
    gender: str
    chief_complaint: str
    vitals: VitalSigns
    has_prior_history: bool = False
    prior_history_summary: Optional[str] = None

class Badge(BaseModel):
    label: str
    type: str

class TriageResult(BaseModel):
    visit_id: str
    recommended_esi: int = Field(..., ge=1, le=5)
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    rationale: str
    badges: List[Badge] = []

class OverrideRequest(BaseModel):
    visit_id: str
    clinician_id: str
    assigned_esi: int = Field(..., ge=1, le=5)
    override_reason: str
