import asyncio
import logging
from datetime import datetime, timezone

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import AuditLog, Visit, as_utc

logger = logging.getLogger(__name__)

# Maximum safe wait by ESI level, in MINUTES (matches the dashboard's deterioration rule).
# MONITOR_SPEEDUP > 1 shrinks them so a demo shows alerts quickly.
ESI_WAIT_MINUTES = {1: 5, 2: 15, 3: 60, 4: 120, 5: 240}


def flag_overdue_visits(now: datetime | None = None) -> int:
    """Marks waiting visits that exceeded their ESI wait limit as NEEDS_REASSESSMENT. Returns how many."""
    now = now or datetime.now(timezone.utc)
    db = SessionLocal()
    flagged = 0
    try:
        for visit in db.query(Visit).filter(Visit.status == "WAITING").all():
            esi = visit.final_esi_score
            if not esi:
                continue
            limit = ESI_WAIT_MINUTES.get(esi, 60) * 60 / max(settings.monitor_speedup, 0.01)
            started = as_utc(visit.last_assessed_at) or as_utc(visit.arrival_time)
            waited = (now - started).total_seconds()  # the clock restarts at every reassessment
            if waited > limit:
                visit.status = "NEEDS_REASSESSMENT"
                db.add(AuditLog(
                    visit_id=visit.id, clinician_id="SYSTEM", action="DETERIORATION_FLAGGED", initial_ai_esi=visit.ai_esi_score,
                    assigned_esi=esi, override_reason=f"Waited {waited / 60:.0f} min (limit {limit / 60:.0f} min for ESI {esi}).",
                ))
                flagged += 1
        if flagged:
            db.commit()
    finally:
        db.close()
    return flagged


async def deterioration_monitor():
    """Background loop: flags patients who have waited past their ESI limit for reassessment."""
    logger.info("Deterioration monitor started")
    while True:
        try:
            n = await asyncio.to_thread(flag_overdue_visits)
            if n:
                logger.warning("Deterioration monitor flagged %d patient(s) for reassessment", n)
        except Exception as e:
            logger.error("Monitor error: %s", e)
        await asyncio.sleep(5)
