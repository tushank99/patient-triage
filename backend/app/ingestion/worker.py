"""
Job runner for the DB-backed queue.

Claiming is a conditional UPDATE (status QUEUED -> RUNNING); only the worker whose update hit one row owns
the job, so several workers (in-process loop, `python -m backend.app.ingestion.worker`, a Render background
worker) can run at once on Postgres or SQLite. RUNNING jobs older than the lease are presumed crashed and requeued.
"""
import asyncio
import logging
import sys
from datetime import timedelta
from typing import Optional

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.ingestion.service import finish_job, process_job
from backend.app.models import IngestionJob, utc_now

logger = logging.getLogger(__name__)


def _recover_stale(db, now) -> int:
    n = (db.query(IngestionJob)
         .filter(IngestionJob.status == "RUNNING", IngestionJob.started_at < now - timedelta(minutes=settings.ingestion_lease_minutes))
         .update({IngestionJob.status: "QUEUED", IngestionJob.last_error: "lease expired (worker crashed?)"}, synchronize_session=False))
    if n:
        db.commit()
    return n


def claim_next(db, now=None, job_id: Optional[str] = None) -> Optional[IngestionJob]:
    """Claims the oldest due job, or exactly `job_id` (Celery messages carry an id). A duplicate/redelivered
    message for a job that is already RUNNING or finished claims nothing."""
    now = now or utc_now()
    _recover_stale(db, now)
    q = db.query(IngestionJob.id).filter(IngestionJob.status == "QUEUED", IngestionJob.next_attempt_at <= now)
    if job_id:
        q = q.filter(IngestionJob.id == job_id)
    candidate = q.order_by(IngestionJob.created_at.asc()).first()
    if candidate is None:
        return None
    won = (db.query(IngestionJob)
           .filter(IngestionJob.id == candidate.id, IngestionJob.status == "QUEUED")
           .update({IngestionJob.status: "RUNNING", IngestionJob.started_at: now, IngestionJob.attempts: IngestionJob.attempts + 1},
                   synchronize_session=False))
    db.commit()
    return db.get(IngestionJob, candidate.id) if won == 1 else None


def run_one(now=None, job_id: Optional[str] = None) -> bool:
    """Claims and runs one job in its own transaction. Returns False when nothing was runnable."""
    from backend.app.ops.observability import JOBS_FINISHED

    db = SessionLocal()
    try:
        job = claim_next(db, now, job_id)
        if job is None:
            return False
        try:
            stats = process_job(db, job)
            finish_job(db, job, None, stats, now)
            db.commit()
            JOBS_FINISHED.labels(job.kind, job.status).inc()
        except Exception as e:  # roll back partial work, then record the failure in a fresh transaction
            db.rollback()
            job = db.get(IngestionJob, job.id)
            logger.warning("Ingestion job %s (%s) failed: %s", job.id, job.kind, e)
            finish_job(db, job, e, None, now)
            db.commit()
            JOBS_FINISHED.labels(job.kind, job.status).inc()
        return True
    finally:
        db.close()


def run_job_by_id(job_id: str) -> bool:
    """Celery entry point. The DB row decides whether the job runs (idempotent under redelivery)."""
    return run_one(job_id=job_id)


def run_stale_queued(older_than_seconds: int = 120) -> int:
    """Celery-mode safety net in the API process: runs jobs whose broker message was evidently lost."""
    db = SessionLocal()
    try:
        cutoff = utc_now() - timedelta(seconds=older_than_seconds)
        ids = [j for (j,) in db.query(IngestionJob.id).filter(IngestionJob.status == "QUEUED", IngestionJob.next_attempt_at <= cutoff).limit(20)]
    finally:
        db.close()
    return sum(1 for j in ids if run_one(job_id=j))


async def safety_net_loop() -> None:
    while True:
        await asyncio.sleep(60)
        try:
            n = await asyncio.to_thread(run_stale_queued)
            if n:
                logger.warning("Safety net ran %d job(s) that Celery never picked up (broker down?)", n)
        except Exception as e:
            logger.error("Safety net error: %s", e)


def run_pending_jobs(max_jobs: int = 1000, now=None) -> int:
    done = 0
    while done < max_jobs and run_one(now):
        done += 1
    return done


async def ingestion_worker_loop() -> None:
    logger.info("Ingestion worker started")
    while True:
        try:
            await asyncio.to_thread(run_pending_jobs, 20)
        except Exception as e:
            logger.error("Ingestion worker error: %s", e)
        await asyncio.sleep(settings.ingestion_poll_seconds)


if __name__ == "__main__":  # standalone worker process
    logging.basicConfig(level=logging.INFO)
    if "--once" in sys.argv:
        print(f"processed {run_pending_jobs()} job(s)")
    else:
        asyncio.run(ingestion_worker_loop())
