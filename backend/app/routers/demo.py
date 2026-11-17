"""Non-destructive demo controls. Nothing here deletes data; ending a surge just closes those visits."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.auth import log_access, require
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import SurgeResponse
from backend.app.services.demo_queue import end_surge, seed_demo_group

router = APIRouter(prefix="/demo", tags=["demo"])


def _demo_only() -> None:
    if not settings.demo_mode:
        raise HTTPException(status_code=404, detail="Not found.")


@router.post("/surge", response_model=SurgeResponse)
def start_surge(user: User = Depends(require("demo:surge")), db: Session = Depends(get_db)):
    _demo_only()
    log_access(db, user, "DEMO_SURGE_START")
    n = seed_demo_group(db, "surge", actor=user.username, facility_id=user.facility_id)
    return SurgeResponse(status="success", message="Mass-casualty surge started.", visits_changed=n)


@router.post("/surge/end", response_model=SurgeResponse)
def stop_surge(user: User = Depends(require("demo:surge")), db: Session = Depends(get_db)):
    _demo_only()
    log_access(db, user, "DEMO_SURGE_END")
    n = end_surge(db, user.facility_id, user.username)
    return SurgeResponse(status="success", message="Surge ended.", visits_changed=n)
