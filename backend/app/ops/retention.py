"""
Retention and erasure enforcement (daily via Celery beat, or `python -m backend.app.ops.retention`).

  * access logs older than RETENTION_ACCESS_LOG_DAYS                      -> deleted
  * finished ingestion jobs older than RETENTION_JOB_DAYS                  -> deleted
  * verified/rejected captures older than RETENTION_CAPTURE_DETAIL_DAYS    -> word timings and OCR boxes stripped
    (the verified text and summary stay)
  * DELETED documents: originals purged, chunks removed, external vector index (Chroma) entries removed. Runs every
    time, so a delete that half-failed (storage or index outage) is completed later: deletion always reaches the index.
  * orphan chunks (document gone or not READY/current) removed from DB and index
  * optional: patients whose last visit ended more than RETENTION_ARCHIVE_ANONYMIZE_DAYS ago are anonymised
Clinical audit logs (decisions, overrides) are NOT deleted here: they follow the facility's medical-records retention.
"""
import json
from datetime import timedelta
from typing import Dict

from sqlalchemy import func

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import (
    ACTIVE_STATUSES, AccessLog, ClinicalDocument, DocumentChunk, IngestionJob, IntakeCapture, Patient, Visit, utc_now,
)


def run_retention(now=None) -> Dict:
    from backend.app.ingestion.service import purge_document, purge_patient_documents
    from backend.app.ingestion.vector_index import get_index
    from backend.app.retrieval.cache import PATIENT_CHUNKS

    now = now or utc_now()
    stats: Dict[str, int] = {}
    db = SessionLocal()
    try:
        stats["access_logs_deleted"] = db.query(AccessLog).filter(
            AccessLog.timestamp < now - timedelta(days=settings.retention_access_log_days)).delete(synchronize_session=False)
        stats["jobs_deleted"] = db.query(IngestionJob).filter(
            IngestionJob.status.in_(("SUCCEEDED", "FAILED", "DEAD")),
            IngestionJob.finished_at < now - timedelta(days=settings.retention_job_days)).delete(synchronize_session=False)

        cutoff = now - timedelta(days=settings.retention_capture_detail_days)
        n = 0
        for c in db.query(IntakeCapture).filter(IntakeCapture.status.in_(("VERIFIED", "REJECTED")), IntakeCapture.verified_at < cutoff):
            if c.output and ("words" in c.output or "pages" in c.output):
                c.output = {"text": c.output.get("text", ""), "detail_purged_at": now.isoformat()}
                n += 1
        stats["captures_detail_purged"] = n

        # Deletion must reach every copy: storage originals, DB chunks and the external vector index.
        n = 0
        for doc in db.query(ClinicalDocument).filter(ClinicalDocument.status == "DELETED"):
            if any(v.storage_key or v.extracted_text for v in doc.versions) or doc.content:
                purge_document(db, doc)
                n += 1
            else:
                get_index().delete_document(db, doc.id)  # idempotent: catches index entries left by a failed earlier purge
        stats["deleted_documents_purged"] = n

        live = db.query(ClinicalDocument.id).filter(ClinicalDocument.status != "DELETED")
        orphans = db.query(DocumentChunk.document_id).filter(~DocumentChunk.document_id.in_(live)).distinct().all()
        for (doc_id,) in orphans:
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete(synchronize_session=False)
            get_index().delete_document(db, doc_id)
        stats["orphan_chunk_documents_removed"] = len(orphans)

        stats["patients_anonymized"] = 0
        if settings.retention_archive_anonymize_days > 0:
            limit = now - timedelta(days=settings.retention_archive_anonymize_days)
            last = (db.query(Visit.patient_id, func.max(Visit.discharge_time)).group_by(Visit.patient_id)
                    .having(func.max(Visit.discharge_time) < limit).all())
            for pid, _ in last:
                if db.query(Visit).filter(Visit.patient_id == pid, Visit.status.in_(ACTIVE_STATUSES)).count():
                    continue
                p = db.get(Patient, pid)
                if p and not p.is_anonymized:
                    import uuid

                    purge_patient_documents(db, p)
                    p.mrn, p.name, p.date_of_birth, p.history_facts, p.is_anonymized = f"REDACTED_{uuid.uuid4().hex[:8]}", "REDACTED", None, None, True
                    stats["patients_anonymized"] += 1
        db.commit()
    finally:
        db.close()
    PATIENT_CHUNKS.clear()  # cached chunk matrices must not outlive the data they came from
    return stats


if __name__ == "__main__":
    print(json.dumps(run_retention(), indent=2))
