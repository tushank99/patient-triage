"""
Celery workers over Redis (JOB_BACKEND=celery).

The ingestion_jobs table stays the single source of truth for job state, attempts, backoff and the dead letter (DEAD).
Celery is only the transport: a message carries a job id; the worker claims that row with the same conditional
UPDATE as the in-process worker, so a duplicated or redelivered message cannot run a job twice. If a message is lost
(Redis restart, broker outage) the beat `sweep` re-dispatches every due QUEUED job, and the API's safety-net loop runs
jobs that stayed QUEUED for minutes. Losing Redis loses no work.

  worker:  celery -A backend.app.ops.celery_app worker -Q ingestion,heavy --concurrency 2
  beat:    celery -A backend.app.ops.celery_app beat
Queues: "ingestion" (parse/chunk/embed/delete), "heavy" (OCR, transcription) so slow work cannot starve indexing.
"""
import logging

from celery import Celery

from backend.app.config import settings

logger = logging.getLogger(__name__)

celery = Celery("triage", broker=settings.redis_url or "memory://", include=["backend.app.ops.celery_app"])
celery.conf.update(
    task_acks_late=True,  # a worker crash mid-job redelivers the message (the DB lease also covers it)
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="ingestion",
    task_routes={"triage.run_ingestion_job": {"queue": "ingestion"}, "triage.run_heavy_job": {"queue": "heavy"}},
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": settings.ingestion_lease_minutes * 60 + 60},
    beat_schedule={
        "sweep-due-jobs": {"task": "triage.sweep", "schedule": 30.0},
        "deterioration-monitor": {"task": "triage.deterioration", "schedule": 10.0},
        "retention-daily": {"task": "triage.retention", "schedule": 24 * 3600.0},
        "backup-daily": {"task": "triage.backup", "schedule": 24 * 3600.0},
    },
    timezone="UTC",
)


@celery.task(name="triage.run_ingestion_job")
def run_ingestion_job(job_id: str) -> bool:
    from backend.app.ingestion.worker import run_job_by_id

    return run_job_by_id(job_id)


@celery.task(name="triage.run_heavy_job")
def run_heavy_job(job_id: str) -> bool:
    from backend.app.ingestion.worker import run_job_by_id

    return run_job_by_id(job_id)


@celery.task(name="triage.sweep")
def sweep() -> int:
    """Re-dispatch due QUEUED jobs whose message may have been lost."""
    from backend.app.database import SessionLocal
    from backend.app.models import IngestionJob, utc_now

    db = SessionLocal()
    try:
        due = db.query(IngestionJob).filter(IngestionJob.status == "QUEUED", IngestionJob.next_attempt_at <= utc_now()).limit(500).all()
        for j in due:
            dispatch(j.id, heavy=j.kind == "INGEST" and _is_heavy(db, j))
        return len(due)
    finally:
        db.close()


@celery.task(name="triage.deterioration")
def deterioration() -> int:
    from backend.app.background_tasks import flag_overdue_visits

    return flag_overdue_visits()


@celery.task(name="triage.retention")
def retention() -> dict:
    from backend.app.ops.retention import run_retention

    return run_retention()


@celery.task(name="triage.backup")
def backup() -> dict:
    from backend.app.ops.backup import create_backup

    return create_backup()


def _is_heavy(db, job) -> bool:
    from backend.app.ingestion.validation import IMAGES
    from backend.app.models import DocumentVersion

    v = db.get(DocumentVersion, job.version_id) if job.version_id else None
    return bool(v and (v.mime_type in IMAGES or (v.mime_type == "application/pdf" and v.parsed_blocks is None)))


def dispatch(job_id: str, heavy: bool = False) -> bool:
    """Send a job id to the broker. Failure is logged, never raised: the sweep/safety net will pick the job up."""
    try:
        (run_heavy_job if heavy else run_ingestion_job).apply_async(args=[job_id], retry=False)
        return True
    except Exception as e:
        logger.warning("Could not dispatch job %s to Celery (%s); it stays QUEUED for the sweep.", job_id, type(e).__name__)
        return False
