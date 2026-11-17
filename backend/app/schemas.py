"""
The API contract: single source of truth for request/response shapes.

contracts/openapi.json is generated from it (python -m backend.scripts.export_openapi) and
frontend/src/lib/api-types.ts from that (npm run gen:api). A test fails if the committed
openapi.json drifts; `tsc` fails if the frontend drifts from api-types.ts.
"""
from datetime import date, datetime
from typing import Any, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Gender = Literal["M", "F", "O"]
BadgeType = Literal["danger", "warning", "success", "default"]
Disposition = Literal["ADMITTED", "DISCHARGED", "LWBS"]
Role = Literal["triage_nurse", "charge_nurse", "physician", "admin", "viewer"]


class StrictModel(BaseModel):
    # Unknown fields are rejected so a frontend sending the wrong shape gets a 422, not a half-parse.
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ auth


class LoginRequest(StrictModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class UserOut(BaseModel):
    username: str
    display_name: str
    role: Role
    facility_id: str
    permissions: List[str]


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserOut


# ------------------------------------------------------------------ intake


class VitalSigns(StrictModel):
    heart_rate: Optional[int] = Field(None, ge=20, le=300)
    systolic_bp: Optional[int] = Field(None, ge=40, le=300)
    diastolic_bp: Optional[int] = Field(None, ge=20, le=200)
    respiratory_rate: Optional[int] = Field(None, ge=4, le=80)
    oxygen_saturation: Optional[float] = Field(None, ge=40.0, le=100.0)
    temperature_celsius: Optional[float] = Field(None, ge=30.0, le=45.0)
    pain_score: Optional[int] = Field(None, ge=0, le=10)


class TriageRequest(StrictModel):
    request_id: UUID = Field(..., description="Client-generated id. Retrying with the same id replays the original result.")
    mrn: Optional[str] = Field(None, description="Existing MRN. Omit for a walk-in; the server assigns one.")
    patient_name: Optional[str] = Field(None, max_length=200, description="Used to scrub the name from text sent to the LLM.")
    age_years: float = Field(..., ge=0.0, le=125.0)
    gender: Gender
    chief_complaint: str = Field(..., min_length=1, max_length=2000)
    vitals: VitalSigns
    transcript_capture_id: Optional[str] = Field(None, description="Set when the complaint came from dictation; it must be confirmed (not PENDING_REVIEW).")


class ReassessRequest(StrictModel):
    request_id: UUID
    vitals: VitalSigns
    chief_complaint: Optional[str] = Field(None, min_length=1, max_length=2000)


# ------------------------------------------------------------------ assessment


class Badge(BaseModel):
    label: str
    type: BadgeType


class RuleHit(BaseModel):
    id: str
    version: int
    esi: int = Field(..., ge=1, le=5)
    reason: str
    source: str


class FeatureContribution(BaseModel):
    """One TreeSHAP term: how much this input pushed the ML model toward its predicted ESI (log-odds)."""

    feature: str
    label: str
    value: Optional[float] = None
    contribution: float


class AssessmentComponents(BaseModel):
    ml_esi: Optional[int] = None
    ml_probability: Optional[float] = None
    ml_drivers: List[FeatureContribution] = []
    llm_esi: Optional[int] = None
    llm_category: Optional[str] = None
    rule_floor_esi: Optional[int] = None
    rules_fired: List[RuleHit] = []
    escalated_by_rules: bool = False
    llm_downgrade_blocked: bool = Field(False, description="True when the LLM suggested a less acute ESI than a fired safety rule allows.")


class ConfidenceBreakdown(BaseModel):
    """Three separate signals. They are never blended into one number."""

    model_config = ConfigDict(protected_namespaces=())

    model_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-class probability of the ML model (not calibrated yet).")
    data_completeness: float = Field(..., ge=0.0, le=1.0, description="Weighted share of vitals actually measured; capped at 0.5 if complaint-critical vitals are missing.")
    retrieval_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="How much of the patient's history was available; null when there is no record to retrieve.")
    retrieval_note: str
    missing_critical: List[str] = []
    model_llm_agree: Optional[bool] = None


class DecisionStep(BaseModel):
    stage: str
    esi_before: Optional[int] = None
    esi_after: int
    note: str


class HistoryDocumentRef(BaseModel):
    id: str
    doc_type: str
    title: str
    doc_date: Optional[date] = None
    snippet: str
    section: Optional[str] = None
    page: Optional[int] = None


class HistoryContext(BaseModel):
    has_history: bool
    documents: List[HistoryDocumentRef] = []
    cited_document_ids: List[str] = []
    retrieval_method: str = "none"
    low_confidence: bool = Field(False, description="True when no history passage matched the complaint confidently; only allergies, medications and problem list were used.")
    top_relevance: Optional[float] = None
    relevance_threshold: Optional[float] = None


class Versions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    pipeline: str
    model: Optional[str] = None
    feature_schema: str
    rules: str
    model_label_source: Optional[str] = Field(None, description="Where the ML training labels came from, e.g. 'synthetic-rules' (demo) or 'mimic-iv-ed'.")
    prompt: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class ClaimOut(BaseModel):
    text: str
    kind: str
    status: str
    citations: List[str] = []
    document_ids: List[str] = []
    verdict: Literal["supported", "partial", "unsupported", "uncited", "invalid_citation"]
    coverage: float = 0.0
    reasons: List[str] = []


class GroundingSummary(BaseModel):
    n_claims: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    uncited: int = 0
    invalid_citation: int = 0
    score: Optional[float] = Field(None, description="Share of the LLM's claims fully supported by the passages they cite.")


class ConditionOut(BaseModel):
    name: str
    status: str
    effective_status: str
    since: Optional[str] = None
    resolved: Optional[str] = None
    note: Optional[str] = None
    source: Optional[str] = None


class MedicationOut(BaseModel):
    name: str
    dose: str = ""
    drug_class: str = ""
    status: str
    effective_status: str
    since: Optional[str] = None
    stopped: Optional[str] = None
    source: Optional[str] = None


class AllergyOut(BaseModel):
    substance: str
    reaction: str = ""
    severity: str = ""
    source: Optional[str] = None


class ConflictOut(BaseModel):
    kind: str
    severity: Literal["high", "moderate"]
    description: str
    sources: List[str] = []


class ClinicalFactsOut(BaseModel):
    conditions: List[ConditionOut] = []
    medications: List[MedicationOut] = []
    allergies: List[AllergyOut] = []
    nkda_statements: List[str] = []
    conflicts: List[ConflictOut] = Field([], description="History-internal and intake-vs-history conflicts.")


class TriageResponse(BaseModel):
    visit_id: str
    mrn: str
    is_new_patient: bool
    recommended_esi: int = Field(..., ge=1, le=5)
    final_esi: int = Field(..., ge=1, le=5, description="ESI currently assigned to the visit.")
    requires_confirmation: bool = Field(False, description="True when the AI suggests a downgrade; a clinician must confirm via /api/override.")
    requires_clinician_review: bool
    review_reasons: List[str] = []
    confidence: ConfidenceBreakdown
    rationale: str
    risk_factors: List[str] = []
    badges: List[Badge] = []
    components: AssessmentComponents
    decision_trace: List[DecisionStep] = []
    history: HistoryContext
    degraded: List[str] = Field([], description="Pipeline stages that were unavailable, e.g. 'llm_unavailable'.")
    warnings: List[str] = []
    pii_redactions: int = 0
    versions: Versions
    claims: List[ClaimOut] = []
    grounding: GroundingSummary = GroundingSummary()
    facts: ClinicalFactsOut = ClinicalFactsOut()
    security_flags: List[str] = Field([], description="e.g. 'prompt_injection_suspected:D2' when instruction-like text was withheld.")


# ------------------------------------------------------------------ patients, queue, archive


class PatientLookup(BaseModel):
    mrn: str
    name: Optional[str] = None
    gender: Gender
    date_of_birth: Optional[date] = None
    age_years: float
    has_history: bool
    document_count: int


class OverrideRequest(StrictModel):
    visit_id: str
    assigned_esi: int = Field(..., ge=1, le=5)
    override_reason: str = Field(..., min_length=3, max_length=1000)


class OverrideResponse(BaseModel):
    status: Literal["success"]
    message: str
    final_esi: int
    below_rule_floor: bool


class ReviewRequest(StrictModel):
    note: Optional[str] = Field(None, max_length=1000)


class DispositionRequest(StrictModel):
    visit_id: str
    disposition: Disposition
    checklist: Optional[dict[str, bool]] = None


class RevertRequest(StrictModel):
    visit_id: str


class AnonymizeRequest(StrictModel):
    mrn: str


class StatusResponse(BaseModel):
    status: Literal["success"]
    message: str


class QueueItem(BaseModel):
    visit_id: str
    patient_mrn: str
    patient_name: Optional[str] = Field(None, description="Null for roles without patient-level access.")
    age: float
    gender: Gender
    complaint: str
    esi_score: Optional[int]
    ai_esi_score: Optional[int]
    ai_model_probability: Optional[float]
    status: str
    wait_time_minutes: float
    arrival_time: datetime
    is_overridden: bool
    needs_review: bool
    review_reasons: List[str] = []
    is_demo_surge: bool = False


class ArchiveItem(BaseModel):
    visit_id: str
    patient_mrn: str
    patient_name: Optional[str]
    age: float
    gender: Gender
    complaint: str
    esi_score: Optional[int]
    ai_esi_score: Optional[int]
    ai_model_probability: Optional[float]
    disposition: str
    arrival_time: datetime
    disposition_time: Optional[datetime]
    is_overridden: bool
    is_anonymized: bool


class VitalsReadingOut(BaseModel):
    recorded_at: datetime
    recorded_by: str
    source: str
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    temperature_celsius: Optional[float] = None
    pain_score: Optional[int] = None
    ai_esi_after: Optional[int] = None


class AuditEventOut(BaseModel):
    timestamp: datetime
    actor: str
    action: str
    initial_ai_esi: Optional[int]
    assigned_esi: int
    reason: Optional[str]
    detail: Optional[dict[str, Any]] = None


class SurgeResponse(BaseModel):
    status: Literal["success"]
    message: str
    visits_changed: int


# ------------------------------------------------------------------ documents & ingestion (Phase 3)

DocType = Literal[
    "discharge_summary", "er_note", "clinic_note", "lab_summary", "imaging_report", "immunization_record",
    "medication_list", "allergy_record", "problem_list", "other",
]


class DocumentOut(BaseModel):
    id: str
    title: str
    doc_type: str
    doc_date: Optional[date] = None
    status: str
    current_version: int
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    is_synthetic: bool
    uploaded_by: Optional[str] = None
    updated_at: datetime
    chunk_count: int = 0
    latest_job_status: Optional[str] = None
    latest_job_error: Optional[str] = None


class DocumentVersionOut(BaseModel):
    version: int
    checksum: str
    size_bytes: int
    mime_type: str
    scan_result: str
    parser: Optional[str] = None
    page_count: Optional[int] = None
    chunker: Optional[str] = None
    embedding_model: Optional[str] = None
    indexed_at: Optional[datetime] = None
    created_by: str
    created_at: datetime
    original_available: bool


class IngestionJobOut(BaseModel):
    id: str
    kind: str
    document_id: str
    status: str
    attempts: int
    max_attempts: int
    last_error: Optional[str] = None
    stats: Optional[dict] = None
    created_by: str
    created_at: datetime
    finished_at: Optional[datetime] = None


class DocumentDetail(DocumentOut):
    versions: List[DocumentVersionOut] = []
    jobs: List[IngestionJobOut] = []


class ChunkOut(BaseModel):
    id: str
    chunk_index: int
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: int
    char_end: int
    text: str
    embedding_model: Optional[str] = None


class UploadResponse(BaseModel):
    status: Literal["created", "new_version", "duplicate"]
    document: DocumentOut
    job_id: Optional[str] = None


class ReindexRequest(StrictModel):
    scope: Literal["stale", "all"] = "stale"


class ReindexResponse(BaseModel):
    status: Literal["success"]
    jobs_enqueued: int


# ------------------------------------------------------------------ multimodal intake (Phase 6)


class CaptureOut(BaseModel):
    id: str
    kind: Literal["transcript", "ocr"]
    status: Literal["PENDING_REVIEW", "AUTO_ACCEPTED", "VERIFIED", "REJECTED"]
    engine: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    low_confidence_count: int
    requires_review: bool
    text: str = Field(..., description="Machine output. Not clinical data until status is VERIFIED or AUTO_ACCEPTED.")
    words: List[dict] = Field([], description="Transcript words with probability, low flag and auto-corrections.")
    corrections: List[dict] = []
    lines: List[dict] = Field([], description="OCR lines with confidence and low flag.")
    tables: List[dict] = []
    fields: List[dict] = []
    verified: Optional[dict] = None
    edit_distance: Optional[int] = None
    document_id: Optional[str] = None
    created_at: datetime
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None


class VerifyCaptureRequest(StrictModel):
    text: Optional[str] = Field(None, max_length=50000, description="The corrected text. Omit to accept the machine text as is.")
    fields: Optional[List[dict]] = None


class RejectCaptureRequest(StrictModel):
    reason: str = Field(..., min_length=3, max_length=500)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[list] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
