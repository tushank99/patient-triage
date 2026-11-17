"""Patient documents: upload (validated), versions, chunks, deletion, and ingestion job monitoring."""
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.auth import authorize_patient, log_access, require
from backend.app.database import get_db
from backend.app.ingestion import service
from backend.app.ingestion.validation import UploadRejected, max_bytes
from backend.app.models import ClinicalDocument, DocumentChunk, IngestionJob, Patient, User
from backend.app.schemas import (
    ChunkOut, DocType, DocumentDetail, DocumentOut, DocumentVersionOut, IngestionJobOut, ReindexRequest, ReindexResponse, UploadResponse,
)

router = APIRouter(tags=["documents"])


def _doc_out(db: Session, doc: ClinicalDocument, job: Optional[IngestionJob] = None) -> DocumentOut:
    chunks = db.query(func.count(DocumentChunk.id)).filter(DocumentChunk.document_id == doc.id).scalar() or 0
    return DocumentOut(
        id=doc.id, title=doc.title, doc_type=doc.doc_type, doc_date=doc.doc_date, status=doc.status,
        current_version=doc.current_version, filename=doc.filename, mime_type=doc.mime_type, is_synthetic=doc.is_synthetic,
        uploaded_by=doc.uploaded_by, updated_at=doc.updated_at, chunk_count=chunks,
        latest_job_status=job.status if job else None, latest_job_error=job.last_error if job else None,
    )


def _job_out(j: IngestionJob) -> IngestionJobOut:
    return IngestionJobOut.model_validate(j, from_attributes=True)


def _patient(db: Session, user: User, mrn: str, action: str) -> Patient:
    return authorize_patient(db, user, db.query(Patient).filter(Patient.mrn == mrn).first(), action)


def _document(db: Session, user: User, document_id: str, action: str) -> ClinicalDocument:
    doc = db.get(ClinicalDocument, document_id)
    if doc is None or doc.patient.is_anonymized:
        raise HTTPException(status_code=404, detail="Document not found.")
    log_access(db, user, action, patient=doc.patient, detail=document_id)
    return doc


async def _read_capped(file: UploadFile) -> bytes:
    data = await file.read(max_bytes() + 1)  # never read more than the limit into memory
    if len(data) > max_bytes():
        raise UploadRejected(413, "File is larger than the upload limit.")
    return data


def _upload_response(db: Session, out: service.UploadOutcome, response: Response) -> UploadResponse:
    db.commit()
    response.status_code = 200 if out.status == "duplicate" else 202
    return UploadResponse(status=out.status, document=_doc_out(db, out.document, out.job), job_id=out.job.id if out.job else None)


@router.post("/patients/{mrn}/documents", response_model=UploadResponse, status_code=202)
async def upload_document(
    mrn: str, response: Response, file: UploadFile = File(...), doc_type: DocType = Form(...), title: Optional[str] = Form(None),
    doc_date: Optional[date] = Form(None), user: User = Depends(require("document:write")), db: Session = Depends(get_db),
):
    patient = _patient(db, user, mrn, "DOCUMENT_UPLOAD")
    try:
        out = service.add_upload(
            db, patient=patient, data=await _read_capped(file), doc_type=doc_type,
            title=(title or file.filename or "Untitled document")[:200], doc_date=doc_date, actor=user.username,
            filename=(file.filename or "")[:200] or None, source="upload",
        )
    except UploadRejected as e:
        log_access(db, user, "DOCUMENT_UPLOAD_REJECTED", patient=patient, allowed=False, detail=e.message)
        db.commit()
        raise HTTPException(status_code=e.status, detail=e.message)
    return _upload_response(db, out, response)


@router.post("/documents/{document_id}/versions", response_model=UploadResponse, status_code=202)
async def upload_new_version(
    document_id: str, response: Response, file: UploadFile = File(...), user: User = Depends(require("document:write")),
    db: Session = Depends(get_db),
):
    doc = _document(db, user, document_id, "DOCUMENT_NEW_VERSION")
    try:
        out = service.add_upload(
            db, patient=doc.patient, data=await _read_capped(file), doc_type=doc.doc_type, title=doc.title, doc_date=doc.doc_date,
            actor=user.username, filename=(file.filename or "")[:200] or None, document_id=doc.id,
        )
    except UploadRejected as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _upload_response(db, out, response)


@router.get("/patients/{mrn}/documents", response_model=List[DocumentOut])
def list_documents(mrn: str, include_deleted: bool = False, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    patient = _patient(db, user, mrn, "DOCUMENT_LIST")
    q = db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == patient.id)
    if not include_deleted:
        q = q.filter(ClinicalDocument.status != "DELETED")
    docs = q.order_by(ClinicalDocument.doc_date.desc().nullslast(), ClinicalDocument.created_at.desc()).all()
    jobs = service.latest_jobs(db, [d.id for d in docs])
    db.commit()
    return [_doc_out(db, d, jobs.get(d.id)) for d in docs]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    doc = _document(db, user, document_id, "DOCUMENT_READ")
    jobs = db.query(IngestionJob).filter(IngestionJob.document_id == doc.id).order_by(IngestionJob.created_at.asc()).all()
    base = _doc_out(db, doc, jobs[-1] if jobs else None)
    db.commit()
    return DocumentDetail(
        **base.model_dump(),
        versions=[DocumentVersionOut(**{k: getattr(v, k) for k in DocumentVersionOut.model_fields if k != "original_available"},
                                     original_available=bool(v.storage_key)) for v in doc.versions],
        jobs=[_job_out(j) for j in jobs],
    )


@router.get("/documents/{document_id}/chunks", response_model=List[ChunkOut])
def get_chunks(document_id: str, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    doc = _document(db, user, document_id, "DOCUMENT_CHUNKS_READ")
    rows = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index).all()
    db.commit()
    return [ChunkOut.model_validate(r, from_attributes=True) for r in rows]


@router.delete("/documents/{document_id}", response_model=IngestionJobOut, status_code=202)
def delete_document(document_id: str, user: User = Depends(require("document:delete")), db: Session = Depends(get_db)):
    doc = _document(db, user, document_id, "DOCUMENT_DELETE")
    if doc.status == "DELETED":
        raise HTTPException(status_code=409, detail="Document is already deleted.")
    job = service.mark_deleted(db, doc, user.username)
    db.commit()
    return _job_out(job)


# ------------------------------------------------------------------ ingestion monitoring


@router.get("/ingestion/jobs", response_model=List[IngestionJobOut])
def list_jobs(status: Optional[str] = None, limit: int = 100, user: User = Depends(require("ingestion:admin")), db: Session = Depends(get_db)):
    q = db.query(IngestionJob)
    if status:
        q = q.filter(IngestionJob.status == status.upper())
    return [_job_out(j) for j in q.order_by(IngestionJob.created_at.desc()).limit(min(limit, 500)).all()]


@router.post("/ingestion/jobs/{job_id}/retry", response_model=IngestionJobOut)
def retry_job(job_id: str, user: User = Depends(require("ingestion:admin")), db: Session = Depends(get_db)):
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status not in ("FAILED", "DEAD"):
        raise HTTPException(status_code=409, detail=f"Only FAILED or DEAD jobs can be retried (this one is {job.status}).")
    service.requeue(db, job)
    log_access(db, user, "INGESTION_JOB_RETRY", detail=job_id)
    db.commit()
    return _job_out(job)


@router.post("/ingestion/reindex", response_model=ReindexResponse)
def reindex(req: ReindexRequest, user: User = Depends(require("ingestion:admin")), db: Session = Depends(get_db)):
    n = service.enqueue_reindex(db, req.scope, user.username)
    log_access(db, user, f"INGESTION_REINDEX_{req.scope.upper()}", detail=f"{n} jobs")
    db.commit()
    return ReindexResponse(status="success", jobs_enqueued=n)


@router.get("/ai/metrics")
def ai_metrics(user: User = Depends(require("ingestion:admin"))) -> dict:
    """LLM observability: calls, failures by reason, latency and tokens per provider/model (recent window, this process)."""
    from backend.app.llm.assessor import metrics_summary

    return metrics_summary()


@router.get("/ingestion/stats")
def ingestion_stats(user: User = Depends(require("ingestion:admin")), db: Session = Depends(get_db)) -> dict:
    return service.stats_summary(db)
