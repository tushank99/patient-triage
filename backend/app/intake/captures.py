"""
Human-verification gate for machine-read input.

  transcript  speech -> text (+ extracted symptoms/duration/severity). AUTO_ACCEPTED only if no word is below the
              confidence threshold, nothing was auto-corrected and no field is low-confidence; otherwise PENDING_REVIEW.
              A triage request that references a PENDING_REVIEW transcript is refused (409).
  ocr         scanned PDF / photo -> text + fields. If ANY line or field is below threshold, the document is held in
              NEEDS_REVIEW: not indexed, not searchable, not used for facts or rules. On verify, the clinician's
              corrected text replaces the OCR text and the document is indexed from it.
The original audio is NOT stored (voice is identifying); only the transcript and word confidences are.
"""
import re
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.intake import extract_speech
from backend.app.intake.ocr import OcrResult
from backend.app.intake.ocr_extract import extract_fields, fields_as_dicts
from backend.app.intake.stt import Transcript, wer
from backend.app.ingestion.parsers import looks_like_heading
from backend.app.models import ClinicalDocument, DocumentVersion, IntakeCapture, utc_now


class CaptureError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def transcript_capture(db: Session, t: Transcript, actor: str, patient_id: Optional[str], request_id: Optional[str]) -> IntakeCapture:
    fields = extract_speech.extract(t.text, t.words, settings.stt_min_word_confidence)
    low_words = t.low_confidence_words
    needs = bool(low_words) or bool(t.corrections) or any(f.low_confidence for f in fields)
    cap = IntakeCapture(
        kind="transcript", status="PENDING_REVIEW" if needs else "AUTO_ACCEPTED", patient_id=patient_id, request_id=request_id,
        engine=t.engine, language=t.language, confidence=t.confidence, low_confidence_count=len(low_words), requires_review=needs,
        output={"text": t.text, "words": t.words, "segments": t.segments, "corrections": t.corrections, "duration_s": t.duration_s},
        fields=extract_speech.fields_as_dicts(fields), created_by=actor,
    )
    db.add(cap)
    db.flush()
    return cap


def ocr_capture(db: Session, result: OcrResult, version: DocumentVersion, actor: str) -> IntakeCapture:
    fields = extract_fields(result)
    low_lines = [l for l in result.lines if l.low]
    needs = bool(low_lines) or any(f.low_confidence for f in fields) or not result.lines
    doc = version.document
    cap = IntakeCapture(
        kind="ocr", status="PENDING_REVIEW" if needs else "AUTO_ACCEPTED", patient_id=doc.patient_id, document_id=doc.id,
        version_id=version.id, engine=result.engine, confidence=result.confidence, low_confidence_count=len(low_lines),
        requires_review=needs, output={"text": result.text, **result.as_dict()}, fields=fields_as_dicts(fields), created_by=actor,
    )
    _identity_check(db, cap, doc)
    db.add(cap)
    db.flush()
    return cap


def _identity_check(db: Session, cap: IntakeCapture, doc: ClinicalDocument) -> None:
    """A scanned page naming a different MRN than the patient it is filed under must be reviewed."""
    for f in cap.fields or []:
        if f["kind"] == "identity" and f["name"] == "mrn" and f["value"] and doc.patient and f["value"].upper() != doc.patient.mrn.upper():
            f["low_confidence"] = True
            f["note"] = f"page shows MRN {f['value']} but the document is filed under {doc.patient.mrn}"
            cap.requires_review, cap.status = True, "PENDING_REVIEW"


def blocks_from_text(text: str, page: Optional[int] = None) -> List[dict]:
    blocks = []
    for line in text.splitlines():
        if line.strip():
            blocks.append({"text": line.strip(), "kind": "heading" if looks_like_heading(line) else "paragraph", "page": page})
    return blocks


def verify(db: Session, cap: IntakeCapture, actor: str, text: Optional[str], fields: Optional[list]) -> IntakeCapture:
    if cap.status not in ("PENDING_REVIEW", "AUTO_ACCEPTED"):
        raise CaptureError(409, f"Capture is already {cap.status}.")
    machine = (cap.output or {}).get("text", "")
    final = machine if text is None else text.strip()
    if not final:
        raise CaptureError(400, "Verified text cannot be empty.")
    cap.verified = {"text": final, "fields": fields if fields is not None else cap.fields}
    cap.edit_distance = wer(machine, final)["errors"]
    cap.status, cap.verified_by, cap.verified_at = "VERIFIED", actor, utc_now()
    if cap.kind == "ocr" and cap.version_id:
        from backend.app.ingestion.service import enqueue

        version = db.get(DocumentVersion, cap.version_id)
        version.parsed_blocks = blocks_from_text(final, 1 if (version.page_count or 1) == 1 else None)
        version.extracted_text = final
        version.parser = (version.parser or "ocr") + "+verified"
        doc = version.document
        if doc.status in ("NEEDS_REVIEW", "READY"):
            doc.status = "PENDING" if doc.current_version == 0 else doc.status
            enqueue(db, "REINDEX", doc.id, version.id, actor)
    return cap


def reject(db: Session, cap: IntakeCapture, actor: str, reason: str) -> IntakeCapture:
    if cap.status == "VERIFIED":
        raise CaptureError(409, "A verified capture cannot be rejected.")
    cap.status, cap.verified_by, cap.verified_at = "REJECTED", actor, utc_now()
    cap.verified = {"rejected_reason": reason}
    if cap.document_id:
        doc = db.get(ClinicalDocument, cap.document_id)
        if doc is not None and doc.status == "NEEDS_REVIEW":
            doc.status = "FAILED"
    return cap


def check_transcript_for_triage(db: Session, capture_id: str) -> IntakeCapture:
    cap = db.get(IntakeCapture, capture_id)
    if cap is None or cap.kind != "transcript":
        raise CaptureError(404, "Transcript not found.")
    if cap.status == "PENDING_REVIEW":
        raise CaptureError(409, "The dictated transcript has low-confidence words that must be confirmed before triage.")
    if cap.status == "REJECTED":
        raise CaptureError(409, "The dictated transcript was rejected.")
    return cap


def word_spans(cap: IntakeCapture) -> List[dict]:
    """For UI highlighting: words with confidence and corrections."""
    return [{k: w.get(k) for k in ("word", "probability", "low", "corrected_from", "char_start", "char_end")}
            for w in (cap.output or {}).get("words", [])]


_SPACES = re.compile(r"\s+")
