"""
Ingestion: upload -> (validate) -> store original -> queue job -> parse -> chunk -> embed -> index.

* Dedupe: sha256 of the raw bytes. The same bytes for the same patient return the existing document.
* Versioning: a new upload to an existing document creates version N+1. The previous version stays
  searchable until the new one is indexed, then its chunks are replaced (incremental, no downtime).
* Stale re-index: each version records its chunker + embedding model; when either changes in config,
  `enqueue_reindex(scope="stale")` re-processes exactly those versions from stored blocks (no re-parse).
* Deletion: chunks disappear from retrieval in the same transaction as the delete request; a DELETE job
  then purges stored originals and any external index, with retries. A metadata tombstone remains.
"""
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.ingestion.chunker import chunk_document, chunker_id
from backend.app.ingestion.embeddings import embedding_text, get_embedder, to_bytes
from backend.app.ingestion.parsers import Block, ParsedDocument, ParseError, parse
from backend.app.ingestion.storage import get_storage, storage_key
from backend.app.ingestion.validation import ValidatedUpload, validate_upload
from backend.app.ingestion.vector_index import get_index
from backend.app.models import ClinicalDocument, DocumentChunk, DocumentVersion, IngestionJob, Patient, utc_now

logger = logging.getLogger(__name__)


class PermanentIngestionError(Exception):
    """Retrying cannot help (corrupt file, missing original). The job fails without retries."""


class NeedsReview(Exception):
    """OCR produced low-confidence output: hold the document until a clinician verifies it."""

    def __init__(self, capture_id: str):
        super().__init__(capture_id)
        self.capture_id = capture_id


def _ocr_version(db: Session, version: DocumentVersion, data: bytes) -> ParsedDocument:
    from backend.app.intake.captures import ocr_capture
    from backend.app.intake.ocr import get_ocr_engine, ocr_document

    if get_ocr_engine() is None:
        raise PermanentIngestionError("This is a scanned/image document and OCR is not available on this server "
                                      "(OCR_ENGINE=none or the OCR packages are not installed).")
    result = ocr_document(data, version.mime_type)
    blocks = [Block(l.text, "heading" if l.heading else "paragraph", l.page) for l in result.lines]
    version.parsed_blocks = [{"text": b.text, "kind": b.kind, "page": b.page} for b in blocks]
    version.extracted_text, version.parser, version.page_count = result.text, f"ocr:{result.engine}", len(result.pages)
    cap = ocr_capture(db, result, version, "OCR")
    if cap.requires_review:
        raise NeedsReview(cap.id)
    return ParsedDocument(blocks, version.parser, version.page_count)


@dataclass
class UploadOutcome:
    status: str  # created | new_version | duplicate
    document: ClinicalDocument
    version: Optional[DocumentVersion]
    job: Optional[IngestionJob]


def db_of(obj) -> Session:
    from sqlalchemy.orm import object_session

    return object_session(obj)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def current_chunker() -> str:
    return chunker_id(settings.chunk_target_chars, settings.chunk_overlap_chars)


def enqueue(db: Session, kind: str, document_id: str, version_id: Optional[str], actor: str) -> IngestionJob:
    job = IngestionJob(kind=kind, document_id=document_id, version_id=version_id, created_by=actor,
                       max_attempts=settings.ingestion_max_attempts, next_attempt_at=utc_now())
    db.add(job)
    db.flush()
    from backend.app.ops.dispatch import schedule

    schedule(db, job.id)  # Celery mode: sent after COMMIT; DB mode: no-op (worker polls)
    return job


# ------------------------------------------------------------------ upload


def add_upload(
    db: Session, *, patient: Patient, data: bytes, doc_type: str, title: str, doc_date: Optional[date], actor: str,
    filename: Optional[str] = None, document_id: Optional[str] = None, source: str = "upload", is_synthetic: bool = False,
    validated: Optional[ValidatedUpload] = None,
) -> UploadOutcome:
    """Validates, dedupes, stores the original and queues an INGEST job. Raises UploadRejected / ValueError."""
    v = validated or validate_upload(data)
    checksum = sha256(v.data)

    doc = db.get(ClinicalDocument, document_id) if document_id else None
    if doc is not None:
        if doc.patient_id != patient.id:
            raise ValueError("Document belongs to a different patient.")
        if doc.status == "DELETED":
            raise ValueError("Document was deleted; upload it as a new document.")
        if doc.checksum == checksum:
            return UploadOutcome("duplicate", doc, None, None)
        status = "new_version"
    else:
        dup = (db.query(ClinicalDocument)
               .filter(ClinicalDocument.patient_id == patient.id, ClinicalDocument.checksum == checksum, ClinicalDocument.status != "DELETED")
               .first())
        if dup is not None:
            return UploadOutcome("duplicate", dup, None, None)
        doc = ClinicalDocument(
            id=document_id or str(uuid.uuid4()), patient_id=patient.id, doc_type=doc_type, title=title, doc_date=doc_date,
            content="", checksum=checksum, source=source, is_synthetic=is_synthetic, status="PENDING", current_version=0,
            filename=filename, mime_type=v.mime_type, uploaded_by=actor,
        )
        db.add(doc)
        db.flush()
        status = "created"

    number = (db.query(func.max(DocumentVersion.version)).filter(DocumentVersion.document_id == doc.id).scalar() or 0) + 1
    key = storage_key(patient.id, doc.id, number, checksum)
    get_storage().put(key, v.data)
    version = DocumentVersion(document_id=doc.id, version=number, checksum=checksum, size_bytes=len(v.data), mime_type=v.mime_type,
                              storage_key=key, scan_result=v.scan_result, created_by=actor)
    db.add(version)
    doc.checksum = checksum  # dedupe against the latest upload
    if status == "new_version":
        doc.title, doc.doc_type = title or doc.title, doc_type or doc.doc_type
        doc.doc_date = doc_date or doc.doc_date
        doc.filename, doc.mime_type = filename or doc.filename, v.mime_type
    db.flush()
    job = enqueue(db, "INGEST", doc.id, version.id, actor)
    return UploadOutcome(status, doc, version, job)


# ------------------------------------------------------------------ processing


def _blocks_of(version: DocumentVersion) -> ParsedDocument:
    if version.parsed_blocks is not None:  # re-index path: no original needed
        return ParsedDocument([Block(b["text"], b["kind"], b.get("page")) for b in version.parsed_blocks], version.parser or "stored", version.page_count)
    if not version.storage_key:
        raise PermanentIngestionError("Original file was purged; cannot parse.")
    try:
        data = get_storage().get(version.storage_key)
    except FileNotFoundError:
        raise PermanentIngestionError("Original file is missing from storage (ephemeral disk?). Re-upload it.")
    from backend.app.ingestion.validation import IMAGES

    if version.mime_type in IMAGES:
        return _ocr_version(db_of(version), version, data)
    try:
        parsed = parse(data, version.mime_type)
    except ParseError as e:
        if version.mime_type == "application/pdf" and "No extractable text" in str(e):
            return _ocr_version(db_of(version), version, data)  # scanned PDF
        raise PermanentIngestionError(str(e)) from e
    version.parsed_blocks = [{"text": b.text, "kind": b.kind, "page": b.page} for b in parsed.blocks]
    version.extracted_text, version.parser, version.page_count = parsed.text, parsed.parser, parsed.page_count
    return parsed


def index_version(db: Session, version: DocumentVersion) -> Dict:
    """Parse (if needed) -> chunk -> embed -> replace the document's chunks with this version's. Returns stats."""
    doc = version.document
    t0 = time.perf_counter()
    try:
        parsed = _blocks_of(version)
    except NeedsReview as e:
        if doc.current_version == 0:
            doc.status = "NEEDS_REVIEW"  # nothing indexed: not searchable, not used for facts, until verified
        return {"needs_review": True, "capture_id": e.capture_id, "parser": version.parser}
    t_parse = time.perf_counter()
    chunks = chunk_document(parsed, settings.chunk_target_chars, settings.chunk_overlap_chars)
    if not chunks:
        raise PermanentIngestionError("Document produced no text chunks.")
    t_chunk = time.perf_counter()
    embedder = get_embedder()
    vectors = embedder.embed([embedding_text(doc.doc_type, c.section, c.text) for c in chunks])
    t_embed = time.perf_counter()

    index = get_index()
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete(synchronize_session=False)
    index.delete_document(db, doc.id)
    rows = [
        DocumentChunk(
            id=f"{version.id}:{c.index}", document_id=doc.id, version_id=version.id, patient_id=doc.patient_id,
            chunk_index=c.index, section=c.section or None, section_index=c.section_index, page_start=c.page_start,
            page_end=c.page_end, char_start=c.char_start, char_end=c.char_end, text=c.text, doc_type=doc.doc_type,
            doc_date=doc.doc_date, embedding=to_bytes(vectors[i]), embedding_model=embedder.name,
        )
        for i, c in enumerate(chunks)
    ]
    db.add_all(rows)
    db.flush()
    index.upsert(db, rows)
    version.chunker, version.embedding_model, version.indexed_at = current_chunker(), embedder.name, utc_now()
    doc.current_version, doc.content, doc.status = version.version, parsed.text, "READY"
    doc.checksum = version.checksum
    t_end = time.perf_counter()
    from backend.app.ops.observability import CHUNKS_INDEXED, DOCS_INDEXED

    DOCS_INDEXED.inc()
    CHUNKS_INDEXED.inc(len(rows))
    return {
        "chunks": len(rows), "pages": parsed.page_count, "chars": len(parsed.text), "embedding_model": embedder.name,
        "vector_backend": index.name, "ms": {"parse": round((t_parse - t0) * 1000, 1), "chunk": round((t_chunk - t_parse) * 1000, 1),
                                               "embed": round((t_embed - t_chunk) * 1000, 1), "write": round((t_end - t_embed) * 1000, 1)},
    }


def mark_deleted(db: Session, doc: ClinicalDocument, actor: str) -> IngestionJob:
    """Immediately removes the document from retrieval; the queued job purges originals/external index."""
    doc.status, doc.deleted_at, doc.deleted_by = "DELETED", utc_now(), actor
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete(synchronize_session=False)
    return enqueue(db, "DELETE", doc.id, None, actor)


def purge_document(db: Session, doc: ClinicalDocument, keep_tombstone: bool = True) -> Dict:
    storage, removed = get_storage(), 0
    for v in doc.versions:
        if v.storage_key:
            storage.delete(v.storage_key)
            removed += 1
        v.storage_key, v.extracted_text, v.parsed_blocks = None, None, None
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete(synchronize_session=False)
    get_index().delete_document(db, doc.id)
    doc.content = ""
    if not keep_tombstone:
        db.delete(doc)
    return {"originals_removed": removed}


def purge_patient_documents(db: Session, patient: Patient) -> int:
    """Used by anonymisation: removes every document, version, chunk and stored original for the patient."""
    docs = db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == patient.id).all()
    for d in docs:
        purge_document(db, d, keep_tombstone=False)
    return len(docs)


def stale_versions(db: Session) -> List[DocumentVersion]:
    """Current versions indexed with a different chunker or embedding model than configured now."""
    embedder = get_embedder()
    return (
        db.query(DocumentVersion).join(ClinicalDocument, ClinicalDocument.id == DocumentVersion.document_id)
        .filter(ClinicalDocument.status == "READY", DocumentVersion.version == ClinicalDocument.current_version,
                (DocumentVersion.chunker != current_chunker()) | (DocumentVersion.embedding_model != embedder.name))
        .all()
    )


def enqueue_reindex(db: Session, scope: str, actor: str) -> int:
    if scope == "all":
        versions = (db.query(DocumentVersion).join(ClinicalDocument, ClinicalDocument.id == DocumentVersion.document_id)
                    .filter(ClinicalDocument.status == "READY", DocumentVersion.version == ClinicalDocument.current_version).all())
    else:
        versions = stale_versions(db)
    for v in versions:
        enqueue(db, "REINDEX", v.document_id, v.id, actor)
    return len(versions)


# ------------------------------------------------------------------ job execution


def process_job(db: Session, job: IngestionJob) -> Dict:
    if job.kind in ("INGEST", "REINDEX"):
        version = db.get(DocumentVersion, job.version_id)
        if version is None or version.document.status == "DELETED":
            return {"skipped": "document deleted"}
        doc = version.document
        if doc.status == "NEEDS_REVIEW" and job.kind == "INGEST" and version.parsed_blocks is not None:
            return {"skipped": "awaiting OCR verification"}
        if job.kind == "INGEST" and doc.current_version > version.version:
            return {"skipped": f"superseded by version {doc.current_version}"}
        if doc.current_version == 0:
            doc.status = "PROCESSING"
        return index_version(db, version)
    if job.kind == "DELETE":
        doc = db.get(ClinicalDocument, job.document_id)
        return purge_document(db, doc) if doc is not None else {"skipped": "already gone"}
    raise PermanentIngestionError(f"Unknown job kind {job.kind}")


def backoff(attempt: int) -> timedelta:
    return timedelta(seconds=min(5 * 2 ** (attempt - 1), 600))


def finish_job(db: Session, job: IngestionJob, error: Optional[BaseException], stats: Optional[Dict], now=None) -> None:
    now = now or utc_now()
    if error is None:
        job.status, job.finished_at, job.stats, job.last_error = "SUCCEEDED", now, stats, None
        return
    job.last_error = f"{type(error).__name__}: {error}"[:2000]
    permanent = isinstance(error, PermanentIngestionError)
    if not permanent and job.attempts < job.max_attempts:
        job.status, job.next_attempt_at = "QUEUED", now + backoff(job.attempts)
        return
    job.status, job.finished_at = ("FAILED" if permanent else "DEAD"), now
    doc = db.get(ClinicalDocument, job.document_id)
    if doc is not None and doc.status in ("PENDING", "PROCESSING") and doc.current_version == 0:
        doc.status = "FAILED"


def requeue(db: Session, job: IngestionJob) -> None:
    job.status, job.attempts, job.next_attempt_at, job.finished_at = "QUEUED", 0, utc_now(), None
    doc = db.get(ClinicalDocument, job.document_id)
    if doc is not None and doc.status == "FAILED":
        doc.status = "PENDING"


def latest_jobs(db: Session, document_ids: List[str]) -> Dict[str, IngestionJob]:
    if not document_ids:
        return {}
    jobs = db.query(IngestionJob).filter(IngestionJob.document_id.in_(document_ids)).order_by(IngestionJob.created_at.asc()).all()
    return {j.document_id: j for j in jobs}


def stats_summary(db: Session) -> Dict:
    by_status = dict(db.query(IngestionJob.status, func.count()).group_by(IngestionJob.status).all())
    docs = dict(db.query(ClinicalDocument.status, func.count()).group_by(ClinicalDocument.status).all())
    recent = db.query(IngestionJob).filter(IngestionJob.status == "SUCCEEDED").order_by(IngestionJob.finished_at.desc()).limit(100).all()
    stages: Dict[str, List[float]] = {}
    for j in recent:
        for k, v in ((j.stats or {}).get("ms") or {}).items():
            stages.setdefault(k, []).append(v)
    return {
        "jobs_by_status": by_status, "documents_by_status": docs,
        "chunks": db.query(func.count(DocumentChunk.id)).scalar() or 0,
        "avg_stage_ms_last_100": {k: round(sum(v) / len(v), 1) for k, v in stages.items()},
        "stale_versions": len(stale_versions(db)),
        "embedding_model": get_embedder().name, "chunker": current_chunker(),
        "vector_backend": get_index().name, "storage_backend": get_storage().name,
    }


def ingest_text_inline(
    db: Session, *, patient: Patient, document_id: str, doc_type: str, title: str, doc_date: Optional[date], text: str,
    actor: str, source: str, is_synthetic: bool,
) -> Tuple[str, Optional[IngestionJob]]:
    """Seeder entry point: the same upload path, for text documents."""
    out = add_upload(db, patient=patient, data=text.encode("utf-8"), doc_type=doc_type, title=title, doc_date=doc_date,
                     actor=actor, document_id=document_id, source=source, is_synthetic=is_synthetic, filename=f"{document_id}.txt")
    return out.status, out.job
