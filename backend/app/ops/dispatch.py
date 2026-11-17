"""
Dispatch-after-commit: jobs are sent to Celery only once the transaction that created them has COMMITTED, so a fast
worker can never look for a job row that does not exist yet (and a rolled-back request never sends a message).
In JOB_BACKEND=db mode this is a no-op: the in-process worker polls the table.
"""
from sqlalchemy import event
from sqlalchemy.orm import Session

from backend.app.config import settings

_KEY = "pending_job_dispatch"


def schedule(db: Session, job_id: str, heavy: bool = False) -> None:
    if settings.job_backend != "celery":
        return
    db.info.setdefault(_KEY, []).append((job_id, heavy))


@event.listens_for(Session, "after_commit")
def _after_commit(session: Session) -> None:
    pending = session.info.pop(_KEY, None)
    if not pending:
        return
    from backend.app.ops.celery_app import dispatch

    for job_id, heavy in pending:
        dispatch(job_id, heavy)


@event.listens_for(Session, "after_rollback")
def _after_rollback(session: Session) -> None:
    session.info.pop(_KEY, None)
