"""Phase 6: speech-to-text and OCR intake, normalisation, extraction, and the human-verification gate."""
import io
import uuid
import wave

import numpy as np
import pytest

from backend.app.intake import extract_speech
from backend.app.intake.normalize import canonical_lab, normalize_quantity, parse_date
from backend.app.intake.ocr import OcrLine, reconstruct_tables
from backend.app.intake.stt import Transcript, _finalise, sniff_audio, wer
from backend.tests.conftest import auth, triage_payload


def wav_bytes(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes((np.sin(np.arange(int(rate * seconds)) / 8) * 3000).astype(np.int16).tobytes())
    return buf.getvalue()


def words(spec):
    return [{"word": (" " if i else "") + w, "start": i * 0.4, "end": i * 0.4 + 0.3, "probability": p} for i, (w, p) in enumerate(spec)]


# ------------------------------------------------------------------ normalisation


@pytest.mark.parametrize("text,iso,conf_min,conf_max", [
    ("Collected: 2025-03-14", "2025-03-14", 0.95, 1.0), ("5 Mar 2024", "2024-03-05", 0.9, 1.0), ("March 5, 2024", "2024-03-05", 0.9, 1.0),
    ("13/04/2025", "2025-04-13", 0.9, 1.0), ("03/04/2025", "2025-03-04", 0.0, 0.5), ("31/02/2025", None, 0.0, 0.0), ("4 Jul 25", "2025-07-04", 0.75, 0.85),
])
def test_dates_are_normalised_and_ambiguity_is_flagged(text, iso, conf_min, conf_max):
    d = parse_date(text, order="MDY")
    assert d.iso == iso and conf_min <= d.confidence <= conf_max, d


def test_units_and_lab_names():
    assert normalize_quantity("temperature", 101.3, "°F").value == 38.5
    assert normalize_quantity("glucose", 180, "mg/dL").value == 10.0 and normalize_quantity("glucose", 180, "mg/dL").unit == "mmol/L"
    assert normalize_quantity("creatinine", 1.2, "mg/dL").value == 106
    assert normalize_quantity("weight", 154, "lbs").value == 69.9
    assert normalize_quantity("potassium", 5.8, "").confidence < 1.0  # missing unit lowers confidence
    assert canonical_lab("Hgb") == "hemoglobin" and canonical_lab("K+") == "potassium" and canonical_lab("Hemoglobin A1c") == "hba1c"


# ------------------------------------------------------------------ speech extraction & correction


def test_speech_extraction_symptoms_negation_duration_severity():
    fs = extract_speech.extract("Sudden crushing chest pain for two hours, eight out of ten, no shortness of breath, I take apixaban, allergic to penicillin.")
    got = {(f.name, f.value) for f in fs}
    assert ("symptom", "chest pain") in got and ("negated_symptom", "shortness of breath") in got
    assert ("duration", "2 hours") in got and ("pain_score", "8") in got and ("onset", "sudden") in got
    assert ("medication", "apixaban") in got and ("allergy", "penicillin") in got
    assert ("medication", "penicillin") not in got  # "allergic to penicillin" is not a medication
    assert {f.value for f in extract_speech.extract("vomiting since last night") if f.name == "duration"} == {"since last night"}


def test_low_confidence_split_drug_name_is_corrected_and_flagged():
    t = _finalise(words([("I", 0.99), ("take", 0.97), ("a", 0.31), ("pixa", 0.28), ("ban", 0.35), ("daily", 0.95)]), [], "en", 3.0, "stub")
    assert "apixaban" in t.text and t.corrections[0]["to"] == "apixaban"
    w = next(x for x in t.words if x["word"] == "apixaban")
    assert w["low"] and w["corrected_from"] == "a pixa ban"
    assert t.text[w["char_start"]:w["char_end"]] == "apixaban"
    clean = _finalise(words([("chest", 0.98), ("pain", 0.97)]), [], "en", 1.0, "stub")
    assert clean.corrections == [] and not clean.low_confidence_words


def test_low_asr_confidence_lowers_field_confidence():
    t = _finalise(words([("pain", 0.95), ("eight", 0.2), ("out", 0.3), ("of", 0.3), ("ten", 0.3)]), [], "en", 2.0, "stub")
    f = next(f for f in extract_speech.extract(t.text, t.words) if f.name == "pain_score")
    assert f.low_confidence and f.confidence < 0.5


def test_wer():
    assert wer("chest pain for two hours", "chest pain for two hours")["wer"] == 0
    assert wer("on apixaban daily", "on a pixel ban daily")["errors"] == 3


def test_audio_sniffing():
    assert sniff_audio(wav_bytes()) == "audio/wav" and sniff_audio(b"\x1aE\xdf\xa3....") == "audio/webm"
    assert sniff_audio(b"hello world") is None


# ------------------------------------------------------------------ OCR


def test_table_reconstruction_from_line_boxes():
    def L(t, x, y):
        return OcrLine(t, 0.95, [x, y, x + 80, y + 20], 1)

    lines = [L("Heading", 10, 0)] + [L(c, x, y) for y, row in ((40, ("Test", "Result", "Units")), (80, ("Potassium", "5.8", "mmol/L")),
                                                            (120, ("Sodium", "139", "mmol/L"))) for x, c in zip((10, 300, 500), row)]
    tables = reconstruct_tables(lines, 1)
    assert len(tables) == 1 and tables[0].rows[1] == ["Potassium", "5.8", "mmol/L"]


def test_real_ocr_reads_a_rendered_lab_report_including_a_sideways_scan():
    pytest.importorskip("rapidocr_onnxruntime")
    import random

    from backend.app.intake.ocr import OcrResult, get_ocr_engine, ocr_image
    from backend.app.intake.ocr_extract import extract_fields
    from backend.eval.ocr_benchmark import degrade, make_document, render

    lines, rows, truth = make_document(0, random.Random(3))
    img, _ = render(lines, rows)
    for variant in ("clean", "rotated 90deg"):
        page = ocr_image(np.array(degrade(img, variant, random.Random(1)))[:, :, ::-1].copy(), 1, get_ocr_engine())
        fields = extract_fields(OcrResult([page], "rapidocr"))
        labs = {f.name: f.normalized_value for f in fields if f.kind == "lab"}
        hits = sum(1 for k, v in truth["labs"].items() if labs.get(k) is not None and abs(labs[k] - v) <= max(0.051, abs(v) * 0.01))
        assert hits >= len(truth["labs"]) - 1, (variant, labs, truth["labs"])
        assert any(f.kind == "identity" and f.value == truth["mrn"] for f in fields)


# ------------------------------------------------------------------ verification gate through the API


class StubTranscriber:
    name = "stub"

    def __init__(self, spec):
        self.spec = spec

    def transcribe(self, data, language=None, prompt=True, correct=True):
        return _finalise(words(self.spec), [], "en", 2.0, "stub")


def _transcribe(client, monkeypatch, spec, request_id=None):
    from backend.app.intake import stt

    monkeypatch.setattr(stt, "get_transcriber", lambda model=None: StubTranscriber(spec))
    return client.post("/api/intake/transcribe", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                       data={"request_id": request_id or uuid.uuid4().hex}, headers=auth(client))


SCEN = {"mrn": None, "age_years": 70, "gender": "F", "complaint": "x", "vitals": {"heart_rate": 80}}


def test_unconfirmed_low_confidence_transcript_blocks_triage_until_verified(client, monkeypatch):
    r = _transcribe(client, monkeypatch, [("chest", 0.98), ("pain", 0.97), ("for", 0.95), ("two", 0.3), ("hours", 0.4)])
    cap = r.json()
    assert r.status_code == 200 and cap["status"] == "PENDING_REVIEW" and cap["low_confidence_count"] == 2
    assert any(f["name"] == "symptom" and f["value"] == "chest pain" for f in cap["fields"])
    body = triage_payload(SCEN, chief_complaint=cap["text"], transcript_capture_id=cap["id"])
    blocked = client.post("/api/triage", json=body, headers=auth(client))
    assert blocked.status_code == 409 and "confirmed" in blocked.json()["error"]["message"]

    v = client.post(f"/api/captures/{cap['id']}/verify", json={"text": "chest pain for three hours"}, headers=auth(client)).json()
    assert v["status"] == "VERIFIED" and v["verified"]["text"] == "chest pain for three hours" and v["edit_distance"] == 1
    body["request_id"] = str(uuid.uuid4())
    assert client.post("/api/triage", json=body, headers=auth(client)).status_code == 200


def test_confident_transcript_is_auto_accepted_and_retries_are_idempotent(client, monkeypatch):
    rid = uuid.uuid4().hex
    first = _transcribe(client, monkeypatch, [("cough", 0.97), ("for", 0.96), ("three", 0.95), ("days", 0.97)], rid).json()
    again = _transcribe(client, monkeypatch, [("different", 0.9)], rid).json()
    assert first["status"] == "AUTO_ACCEPTED" and again["id"] == first["id"] and again["text"] == first["text"]


def test_audio_validation_and_disabled_engine(client, monkeypatch):
    from backend.app.config import settings

    r = client.post("/api/intake/transcribe", files={"file": ("a.wav", b"not audio at all", "audio/wav")}, data={"request_id": uuid.uuid4().hex}, headers=auth(client))
    assert r.status_code == 415
    monkeypatch.setattr(settings, "stt_engine", "none")
    from backend.app.intake import stt

    stt.get_transcriber.cache_clear()
    r = client.post("/api/intake/transcribe", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"request_id": uuid.uuid4().hex}, headers=auth(client))
    assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"
    stt.get_transcriber.cache_clear()


def _png(text_lines):
    from PIL import Image, ImageDraw

    from backend.eval.ocr_benchmark import find_font

    img = Image.new("RGB", (1400, 140 + 70 * len(text_lines)), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(text_lines):
        d.text((80, 60 + 70 * i), t, fill="black", font=find_font(34))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_low_confidence_scan_is_held_until_verified_then_indexed(client, monkeypatch):
    pytest.importorskip("rapidocr_onnxruntime")
    from backend.app.config import settings
    from backend.app.ingestion.worker import run_pending_jobs
    from backend.tests.test_ingestion import new_patient, upload

    new_patient("MRN-TEST-OCR")
    monkeypatch.setattr(settings, "ocr_min_line_confidence", 0.9999)  # force "low confidence"
    doc = upload(client, "MRN-TEST-OCR", _png(["MEDICATIONS", "Warfarin 5 mg daily", "Metoprolol 50 mg daily"]), name="scan.png",
                 doc_type="medication_list").json()["document"]
    run_pending_jobs()
    d = client.get(f"/api/documents/{doc['id']}", headers=auth(client)).json()
    assert d["status"] == "NEEDS_REVIEW" and d["chunk_count"] == 0
    assert client.get("/api/patients/MRN-TEST-OCR", headers=auth(client)).json()["has_history"] is False  # not clinical data yet
    caps = client.get("/api/patients/MRN-TEST-OCR/captures?status=PENDING_REVIEW", headers=auth(client)).json()
    cap = next(c for c in caps if c["document_id"] == doc["id"])
    assert cap["kind"] == "ocr" and cap["lines"] and any("Warfarin" in l["text"] for l in cap["lines"])

    corrected = "MEDICATIONS\nWarfarin 5 mg daily\nMetoprolol 25 mg daily"
    assert client.post(f"/api/captures/{cap['id']}/verify", json={"text": corrected}, headers=auth(client)).json()["status"] == "VERIFIED"
    run_pending_jobs()
    d = client.get(f"/api/documents/{doc['id']}", headers=auth(client)).json()
    assert d["status"] == "READY" and d["chunk_count"] >= 1
    chunks = " ".join(c["text"] for c in client.get(f"/api/documents/{doc['id']}/chunks", headers=auth(client)).json())
    assert "Metoprolol 25 mg" in chunks  # the clinician's correction, not the OCR text, is what gets indexed


def test_confident_scan_is_indexed_and_rejection_fails_a_held_scan(client, monkeypatch):
    pytest.importorskip("rapidocr_onnxruntime")
    from backend.app.config import settings
    from backend.app.ingestion.worker import run_pending_jobs
    from backend.tests.test_ingestion import new_patient, upload

    new_patient("MRN-TEST-OCR2")
    monkeypatch.setattr(settings, "ocr_min_line_confidence", 0.5)
    monkeypatch.setattr(settings, "ocr_min_field_confidence", 0.5)
    ok = upload(client, "MRN-TEST-OCR2", _png(["Clinic note", "Patient doing well after knee surgery."]), name="ok.png").json()["document"]
    run_pending_jobs()
    assert client.get(f"/api/documents/{ok['id']}", headers=auth(client)).json()["status"] == "READY"

    monkeypatch.setattr(settings, "ocr_min_line_confidence", 0.9999)
    held = upload(client, "MRN-TEST-OCR2", _png(["Clinic note", "Follow up in two weeks for wound check."]), name="held.png").json()["document"]
    run_pending_jobs()
    cap = next(c for c in client.get("/api/patients/MRN-TEST-OCR2/captures", headers=auth(client)).json() if c["document_id"] == held["id"])
    assert client.post(f"/api/captures/{cap['id']}/reject", json={"reason": "illegible"}, headers=auth(client)).json()["status"] == "REJECTED"
    assert client.get(f"/api/documents/{held['id']}", headers=auth(client)).json()["status"] == "FAILED"


def test_scan_naming_another_patients_mrn_requires_review(client, monkeypatch):
    pytest.importorskip("rapidocr_onnxruntime")
    from backend.app.config import settings
    from backend.app.ingestion.worker import run_pending_jobs
    from backend.tests.test_ingestion import new_patient, upload

    new_patient("MRN-TEST-OCR3")
    monkeypatch.setattr(settings, "ocr_min_line_confidence", 0.3)
    monkeypatch.setattr(settings, "ocr_min_field_confidence", 0.3)
    doc = upload(client, "MRN-TEST-OCR3", _png(["LAB REPORT", "MRN: MRN-9999", "Potassium 5.8 mmol/L"]), name="other.png", doc_type="lab_summary").json()["document"]
    run_pending_jobs()
    assert client.get(f"/api/documents/{doc['id']}", headers=auth(client)).json()["status"] == "NEEDS_REVIEW"
    cap = next(c for c in client.get("/api/patients/MRN-TEST-OCR3/captures", headers=auth(client)).json() if c["document_id"] == doc["id"])
    assert any("filed under MRN-TEST-OCR3" in (f.get("note") or "") for f in cap["fields"])


def test_image_upload_safety(client, monkeypatch):
    from backend.app.config import settings
    from backend.tests.test_ingestion import new_patient, upload

    new_patient("MRN-TEST-OCR")
    assert upload(client, "MRN-TEST-OCR", b"\x89PNG\r\n\x1a\n" + b"garbage" * 50, name="x.png").status_code == 415
    monkeypatch.setattr(settings, "max_image_pixels", 1000)
    assert upload(client, "MRN-TEST-OCR", _png(["big"]), name="big.png").status_code in (413, 415)
