"""Speech and OCR captures: transcribe dictation, review/verify/reject machine-read input."""
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.auth import authorize_patient, log_access, require
from backend.app.database import get_db
from backend.app.intake.captures import CaptureError, reject, transcript_capture, verify, word_spans
from backend.app.models import IntakeCapture, Patient, User
from backend.app.schemas import CaptureOut, RejectCaptureRequest, VerifyCaptureRequest

router = APIRouter(tags=["intake"])


def _out(c: IntakeCapture) -> CaptureOut:
    o = c.output or {}
    lines = [ln for p in o.get("pages", []) for ln in p.get("lines", [])]
    tables = [t for p in o.get("pages", []) for t in p.get("tables", [])]
    return CaptureOut(
        id=c.id, kind=c.kind, status=c.status, engine=c.engine, language=c.language, confidence=c.confidence,  # type: ignore[arg-type]
        low_confidence_count=c.low_confidence_count, requires_review=c.requires_review, text=o.get("text", ""),
        words=word_spans(c), corrections=o.get("corrections", []), lines=lines, tables=tables, fields=c.fields or [],
        verified=c.verified, edit_distance=c.edit_distance, document_id=c.document_id, created_at=c.created_at,
        verified_by=c.verified_by, verified_at=c.verified_at,
    )


def _capture(db: Session, user: User, capture_id: str, action: str) -> IntakeCapture:
    c = db.get(IntakeCapture, capture_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Capture not found.")
    if c.patient_id:
        authorize_patient(db, user, db.get(Patient, c.patient_id), action)
    else:
        log_access(db, user, action, detail=capture_id)
    return c


@router.post("/intake/transcribe", response_model=CaptureOut)
async def transcribe(
    file: UploadFile = File(...), request_id: str = Form(..., min_length=8, max_length=64), language: Optional[str] = Form(None),
    mrn: Optional[str] = Form(None), user: User = Depends(require("triage:write")), db: Session = Depends(get_db),
):
    """Dictation -> draft transcript + extracted symptoms/duration/severity. Retrying with the same request_id returns
    the same capture (no second transcription). The audio itself is not stored."""
    from backend.app.config import settings
    from backend.app.intake.stt import AudioRejected, get_transcriber, validate_audio

    existing = db.query(IntakeCapture).filter(IntakeCapture.request_id == request_id).first()
    if existing:
        return _out(existing)
    patient_id = None
    if mrn:
        patient_id = authorize_patient(db, user, db.query(Patient).filter(Patient.mrn == mrn).first(), "TRANSCRIBE").id
    data = await file.read(int(settings.max_audio_mb * 1024 * 1024) + 1)
    try:
        validate_audio(data)
        transcriber = get_transcriber()
        if transcriber is None:
            raise HTTPException(status_code=503, detail="Server speech-to-text is disabled (STT_ENGINE=none). Use browser dictation or type.")
        t = transcriber.transcribe(data, language or None)
    except AudioRejected as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    if not t.text:
        raise HTTPException(status_code=422, detail="No speech was detected in the recording.")
    cap = transcript_capture(db, t, user.username, patient_id, request_id)
    log_access(db, user, "TRANSCRIBE", detail=cap.id)
    db.commit()
    return _out(cap)


@router.get("/captures/{capture_id}", response_model=CaptureOut)
def get_capture(capture_id: str, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    c = _capture(db, user, capture_id, "CAPTURE_READ")
    db.commit()
    return _out(c)


@router.get("/patients/{mrn}/captures", response_model=List[CaptureOut])
def list_captures(mrn: str, status: Optional[str] = None, user: User = Depends(require("patient:read")), db: Session = Depends(get_db)):
    p = authorize_patient(db, user, db.query(Patient).filter(Patient.mrn == mrn).first(), "CAPTURE_LIST")
    q = db.query(IntakeCapture).filter(IntakeCapture.patient_id == p.id)
    if status:
        q = q.filter(IntakeCapture.status == status.upper())
    rows = q.order_by(IntakeCapture.created_at.desc()).all()
    db.commit()
    return [_out(c) for c in rows]


@router.post("/captures/{capture_id}/verify", response_model=CaptureOut)
def verify_capture(capture_id: str, req: VerifyCaptureRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db)):
    """A clinician confirms (optionally corrects) the machine output. Only now does it become clinical data."""
    c = _capture(db, user, capture_id, "CAPTURE_VERIFY")
    try:
        verify(db, c, user.username, req.text, req.fields)
    except CaptureError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    db.commit()
    return _out(c)


@router.post("/captures/{capture_id}/reject", response_model=CaptureOut)
def reject_capture(capture_id: str, req: RejectCaptureRequest, user: User = Depends(require("triage:write")), db: Session = Depends(get_db)):
    c = _capture(db, user, capture_id, "CAPTURE_REJECT")
    try:
        reject(db, c, user.username, req.reason)
    except CaptureError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    db.commit()
    return _out(c)
