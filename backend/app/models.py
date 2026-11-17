import uuid
from datetime import date, datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on read; treat naive datetimes as UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# Visit statuses that mean "still in the waiting room".
ACTIVE_STATUSES = ("WAITING", "NEEDS_REASSESSMENT")
# Terminal statuses that are not clinical dispositions (excluded from the archive).
NON_ARCHIVE_STATUSES = ("SURGE_ENDED",)

ROLES = ("triage_nurse", "charge_nurse", "physician", "admin", "viewer")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    facility_id: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AccessLog(Base):
    """Who looked at what (HIPAA-style access audit). Separate from AuditLog, which records clinical decisions."""

    __tablename__ = "access_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    username: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str | None] = mapped_column(String, nullable=True)
    action: Mapped[str] = mapped_column(String)  # LOGIN, LOGIN_FAILED, PATIENT_LOOKUP, VISIT_READ, ...
    patient_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    visit_id: Mapped[str | None] = mapped_column(String, nullable=True)
    allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mrn: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    age_years: Mapped[float] = mapped_column(Float, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)  # "M" | "F" | "O"
    # Structured facts (conditions, medications, allergies). "Has history" is computed from documents.
    history_facts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_anonymized: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    visits = relationship("Visit", back_populates="patient")
    documents = relationship("ClinicalDocument", back_populates="patient", cascade="all, delete-orphan")


DOC_STATUSES = ("PENDING", "PROCESSING", "NEEDS_REVIEW", "READY", "FAILED", "DELETED")
JOB_STATUSES = ("QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "DEAD")


class ClinicalDocument(Base):
    """A logical patient document. Its content changes through DocumentVersion rows; only the current
    version is chunked and indexed. `content` caches the current version's extracted text."""

    __tablename__ = "clinical_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), index=True, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    doc_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    checksum: Mapped[str] = mapped_column(String, nullable=False)  # sha256 of the current version's bytes
    source: Mapped[str] = mapped_column(String, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="READY", server_default="READY", index=True)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    filename: Mapped[str | None] = mapped_column(String, nullable=True)  # display only; never used as a path
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    uploaded_by: Mapped[str | None] = mapped_column(String, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    patient = relationship("Patient", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan", order_by="DocumentVersion.version")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", passive_deletes=True)


class DocumentVersion(Base):
    """One uploaded revision. The original bytes live in object storage under `storage_key`;
    the extracted text is kept here so re-chunking/re-embedding never needs the original."""

    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String, ForeignKey("clinical_documents.id", ondelete="CASCADE"), index=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String, index=True, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String, nullable=True)  # None once purged
    scan_result: Mapped[str] = mapped_column(String, nullable=False, default="not_scanned")
    parser: Mapped[str | None] = mapped_column(String, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_blocks: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [{text, kind, page}]: re-chunk without re-parsing
    chunker: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "section-recursive-v1:1000/150"
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    document = relationship("ClinicalDocument", back_populates="versions")


class DocumentChunk(Base):
    """Retrieval unit. patient_id is denormalised so every vector query can be patient-filtered in SQL."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # f"{version_id}:{chunk_index}"
    document_id: Mapped[str] = mapped_column(String, ForeignKey("clinical_documents.id", ondelete="CASCADE"), index=True, nullable=False)
    version_id: Mapped[str] = mapped_column(String, ForeignKey("document_versions.id", ondelete="CASCADE"), index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    section_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    doc_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # float32, L2-normalised
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    document = relationship("ClinicalDocument", back_populates="chunks")


class IngestionJob(Base):
    """Database-backed job queue: works on Postgres and SQLite, survives restarts, needs no Redis."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kind: Mapped[str] = mapped_column(String, nullable=False)  # INGEST | REINDEX | DELETE
    document_id: Mapped[str] = mapped_column(String, index=True, nullable=False)  # no FK: DELETE jobs outlive their document rows' chunks
    version_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="QUEUED", index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"), nullable=False)
    facility_id: Mapped[str] = mapped_column(String, index=True, nullable=False, default="ED-MAIN")
    arrival_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_assessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    request_id: Mapped[str | None] = mapped_column(String, unique=True, index=True, nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    last_reassess_request_id: Mapped[str | None] = mapped_column(String, nullable=True)

    chief_complaint: Mapped[str] = mapped_column(Text, nullable=False)
    heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    systolic_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diastolic_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    respiratory_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    oxygen_saturation: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_celsius: Mapped[float | None] = mapped_column(Float, nullable=True)
    pain_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ai_esi_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # model probability, 0..1
    ai_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_risk_factors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ai_assessment: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # full response, replayed on retries

    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    final_esi_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="WAITING")
    is_overridden: Mapped[bool] = mapped_column(Boolean, default=False)
    is_demo_surge: Mapped[bool] = mapped_column(Boolean, default=False)
    discharge_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient = relationship("Patient", back_populates="visits")
    audit_logs = relationship("AuditLog", back_populates="visit")
    vitals_readings = relationship("VitalsReading", back_populates="visit", order_by="VitalsReading.recorded_at")


class VitalsReading(Base):
    """Every set of vitals submitted for a visit (intake and each re-triage), for the trend view."""

    __tablename__ = "vitals_readings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    visit_id: Mapped[str] = mapped_column(String, ForeignKey("visits.id"), index=True, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    recorded_by: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)  # INTAKE | REASSESS | SEED
    heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    systolic_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diastolic_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    respiratory_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    oxygen_saturation: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_celsius: Mapped[float | None] = mapped_column(Float, nullable=True)
    pain_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_esi_after: Mapped[int | None] = mapped_column(Integer, nullable=True)

    visit = relationship("Visit", back_populates="vitals_readings")


class AuditLog(Base):
    """Clinical decision trail: AI assessments, overrides, reviews, dispositions, reversions."""

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


CAPTURE_STATUSES = ("PENDING_REVIEW", "AUTO_ACCEPTED", "VERIFIED", "REJECTED")


class IntakeCapture(Base):
    """Machine-read input (speech transcript or OCR of a scanned document) waiting for, or past, human verification.
    Nothing in `output`/`fields` is clinical data until status is VERIFIED; `verified` holds what the clinician confirmed."""

    __tablename__ = "intake_captures"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kind: Mapped[str] = mapped_column(String, nullable=False)  # transcript | ocr
    status: Mapped[str] = mapped_column(String, nullable=False, default="PENDING_REVIEW", index=True)  # PENDING_REVIEW | AUTO_ACCEPTED | VERIFIED | REJECTED
    patient_id: Mapped[str | None] = mapped_column(String, ForeignKey("patients.id"), index=True, nullable=True)
    document_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    version_id: Mapped[str | None] = mapped_column(String, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)  # idempotent retries of an upload
    engine: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # overall (mean token/line confidence)
    low_confidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    output: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # transcript segments/words or OCR lines/tables
    fields: Mapped[list | None] = mapped_column(JSON, nullable=True)  # extracted structured fields with confidence
    verified: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {text, fields} as confirmed by a clinician
    edit_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)  # how much the clinician changed (correction metric)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    verified_by: Mapped[str | None] = mapped_column(String, nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
