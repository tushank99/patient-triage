"""Phase 3: upload validation, parsing, chunking, indexing, versioning, deletion, background jobs, migrations."""
import io
import zipfile
from datetime import timedelta

import numpy as np
import pytest

from backend.app.ingestion import service
from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.embeddings import HashingEmbedder
from backend.app.ingestion.parsers import Block, ParsedDocument, parse
from backend.app.ingestion.validation import DOCX, PDF, TEXT, UploadRejected, validate_upload
from backend.app.ingestion.worker import run_pending_jobs
from backend.app.models import utc_now
from backend.tests.conftest import auth, triage_payload

fitz = pytest.importorskip("fitz")
docx = pytest.importorskip("docx")

NOTE = """DISCHARGE SUMMARY

History of Present Illness:
Patient admitted with community acquired pneumonia. Treated with ceftriaxone and azithromycin. Improved by day three.

Medications:
- Warfarin 5 mg daily (anticoagulant), INR goal 2-3
- Metoprolol 50 mg daily

Allergies:
Penicillin causes hives.
"""


# ------------------------------------------------------------------ builders


def make_pdf(pages, encrypt=False) -> bytes:
    doc = fitz.open()
    for items in pages:
        page = doc.new_page()
        y = 72
        for text, size in items:
            page.insert_text((72, y), text, fontsize=size)
            y += size * 2
    kw = {"encryption": fitz.PDF_ENCRYPT_AES_256, "owner_pw": "o", "user_pw": "u"} if encrypt else {}
    return doc.tobytes(**kw)


def make_docx(macro=False) -> bytes:
    d = docx.Document()
    d.add_heading("Medications", level=1)
    d.add_paragraph("Apixaban 5 mg twice daily for atrial fibrillation.")
    d.add_heading("Labs", level=1)
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "INR", "2.6"
    t.cell(1, 0).text, t.cell(1, 1).text = "Creatinine", "1.1"
    buf = io.BytesIO()
    d.save(buf)
    if not macro:
        return buf.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.writestr("word/vbaProject.bin", b"\x00" * 64)
    return out.getvalue()


def new_patient(mrn: str):
    from backend.app.database import SessionLocal
    from backend.app.models import Patient

    db = SessionLocal()
    if not db.query(Patient).filter(Patient.mrn == mrn).first():
        db.add(Patient(mrn=mrn, name="Doc, Test", age_years=70, gender="F"))
        db.commit()
    db.close()


def upload(client, mrn, data: bytes, name="note.txt", doc_type="clinic_note", user="lee", **form):
    return client.post(f"/api/patients/{mrn}/documents", files={"file": (name, data, "application/octet-stream")},
                       data={"doc_type": doc_type, **form}, headers=auth(client, user))


# ------------------------------------------------------------------ validation


def test_type_is_decided_by_bytes_not_by_name():
    assert validate_upload(make_pdf([[("Hello", 11)]])).mime_type == PDF
    assert validate_upload(make_docx()).mime_type == DOCX
    assert validate_upload(NOTE.encode()).mime_type == TEXT
    with pytest.raises(UploadRejected) as e:
        validate_upload(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)  # an image named report.pdf is still an image
    assert e.value.status == 415


@pytest.mark.parametrize("payload,reason", [
    (lambda: make_pdf([[("secret", 11)]], encrypt=True), "Encrypted"),
    (lambda: (lambda b: b.split(b"\n", 1)[0] + b"\n% /JavaScript (app.alert(1))\n" + b.split(b"\n", 1)[1])(make_pdf([[("x", 11)]])), "active content"),
    (lambda: make_docx(macro=True), "Macro-enabled"),
    (lambda: b"PK\x03\x04garbage", "corrupt"),
    (lambda: b"", "Empty"),
])
def test_dangerous_or_broken_files_are_rejected(payload, reason):
    with pytest.raises(UploadRejected) as e:
        validate_upload(payload())
    assert reason.lower() in e.value.message.lower()


def test_zip_bomb_docx_is_rejected():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/media/bomb.bin", b"\x00" * (60 * 1024 * 1024))
    with pytest.raises(UploadRejected, match="zip bomb"):
        validate_upload(out.getvalue())


def test_upload_limits_and_errors_over_http(client, monkeypatch):
    from backend.app.config import settings

    new_patient("MRN-TEST-DOCS")
    monkeypatch.setattr(settings, "max_upload_mb", 0.001)
    r = upload(client, "MRN-TEST-DOCS", b"x" * 5000)
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    monkeypatch.setattr(settings, "max_upload_mb", 10.0)
    r = upload(client, "MRN-TEST-DOCS", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, name="scan.pdf")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"
    assert upload(client, "MRN-TEST-DOCS", NOTE.encode(), doc_type="not_a_type").status_code == 422


# ------------------------------------------------------------------ parsing & chunking


def test_pdf_parser_keeps_headings_pages_and_reading_order():
    pdf = make_pdf([
        [("Discharge Summary", 18), ("Medications", 15), ("Warfarin 5 mg daily.", 11)],
        [("Allergies", 15), ("Penicillin causes hives.", 11)],
    ])
    p = parse(pdf, PDF)
    assert p.page_count == 2
    headings = [b.text for b in p.blocks if b.kind == "heading"]
    assert headings == ["Discharge Summary", "Medications", "Allergies"]
    assert [b.page for b in p.blocks if "Penicillin" in b.text] == [2]


def test_image_only_pdf_fails_with_a_clear_message():
    from backend.app.ingestion.parsers import ParseError

    doc = fitz.open()
    doc.new_page()
    with pytest.raises(ParseError, match="scanned"):
        parse(doc.tobytes(), PDF)


def test_docx_parser_reads_headings_and_tables_in_order():
    p = parse(make_docx(), DOCX)
    assert [(b.kind, b.text) for b in p.blocks] == [
        ("heading", "Medications"), ("paragraph", "Apixaban 5 mg twice daily for atrial fibrillation."),
        ("heading", "Labs"), ("table", "INR | 2.6"), ("table", "Creatinine | 1.1"),
    ]


def test_text_parser_detects_clinical_headings():
    headings = [b.text for b in parse(NOTE.encode(), TEXT).blocks if b.kind == "heading"]
    assert headings == ["DISCHARGE SUMMARY", "History of Present Illness", "Medications", "Allergies"]


def test_chunks_never_cross_sections_overlap_and_track_provenance():
    long = " ".join(f"Sentence number {i} about the hospital course and treatment response." for i in range(40))
    doc = ParsedDocument([
        Block("Hospital Course", "heading", 1), Block(long, "paragraph", 1),
        Block("Allergies", "heading", 2), Block("Penicillin causes hives.", "paragraph", 2),
    ], "test")
    chunks = chunk_document(doc, target=300, overlap=80)
    course = [c for c in chunks if c.section == "Hospital Course"]
    allergy = [c for c in chunks if c.section == "Allergies"]
    assert len(course) > 3 and len(allergy) == 1 and "Penicillin" not in " ".join(c.text for c in course)
    assert all(len(c.text) <= 300 for c in chunks)
    for a, b in zip(course, course[1:]):  # consecutive chunks share text
        assert a.text.split(". ")[-1][:30] in b.text
    assert allergy[0].page_start == allergy[0].page_end == 2
    text = doc.text
    for c in chunks:  # offsets point back into the extracted text
        assert text[c.char_start:c.char_end].split()[:3] == c.text.split()[:3]
    assert chunk_document(doc, 300, 80) == chunks  # deterministic


def test_hashing_embedder_is_deterministic_normalised_and_lexically_sensible():
    e = HashingEmbedder()
    v = e.embed(["warfarin anticoagulant fall", "warfarin anticoagulant fall", "on warfarin, fell at home", "sprained ankle"])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(v[0], v[1])
    assert v[0] @ v[2] > v[0] @ v[3]


# ------------------------------------------------------------------ end to end


def test_upload_is_processed_in_the_background_and_reaches_triage(client):
    new_patient("MRN-TEST-DOCS")
    r = upload(client, "MRN-TEST-DOCS", NOTE.encode(), name="discharge.txt", doc_type="discharge_summary", doc_date="2025-03-01")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "created" and body["job_id"] and body["document"]["status"] == "PENDING"
    doc_id = body["document"]["id"]

    assert run_pending_jobs() >= 1
    detail = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()
    assert detail["status"] == "READY" and detail["current_version"] == 1 and detail["chunk_count"] >= 2
    v = detail["versions"][0]
    assert v["parser"].startswith("text") and v["embedding_model"] == "hash-v2" and v["original_available"] and v["scan_result"] == "not_scanned"
    assert detail["jobs"][-1]["status"] == "SUCCEEDED" and detail["jobs"][-1]["stats"]["chunks"] == detail["chunk_count"]

    chunks = client.get(f"/api/documents/{doc_id}/chunks", headers=auth(client)).json()
    assert {"Medications", "Allergies"} <= {c["section"] for c in chunks}

    from backend.app.database import SessionLocal
    from backend.app.models import DocumentChunk, Patient

    db = SessionLocal()
    pid = db.query(Patient).filter(Patient.mrn == "MRN-TEST-DOCS").one().id
    rows = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).all()
    assert all(c.patient_id == pid and c.doc_date.isoformat() == "2025-03-01" and c.embedding for c in rows)
    db.close()

    scen = {"mrn": "MRN-TEST-DOCS", "age_years": 70, "gender": "F", "complaint": "pneumonia follow up, cough",
            "vitals": {"heart_rate": 90, "oxygen_saturation": 96.0}}
    res = client.post("/api/triage", json=triage_payload(scen), headers=auth(client)).json()
    assert res["history"]["has_history"] and doc_id in {d["id"] for d in res["history"]["documents"]}
    snippet = next(d["snippet"] for d in res["history"]["documents"] if d["id"] == doc_id)
    assert "pneumonia" in snippet.lower()  # best-matching chunk, not the start of the document


def test_retrieval_is_strictly_patient_scoped(client):
    new_patient("MRN-TEST-OTHER")
    upload(client, "MRN-TEST-OTHER", b"Clinic note.\n\nPatient has zebrafishosis, a very rare condition.", name="z.txt")
    run_pending_jobs()
    scen = {"mrn": "MRN-TEST-DOCS", "age_years": 70, "gender": "F", "complaint": "zebrafishosis flare", "vitals": {"heart_rate": 80}}
    res = client.post("/api/triage", json=triage_payload(scen), headers=auth(client)).json()
    assert "zebrafishosis" not in " ".join(d["snippet"] for d in res["history"]["documents"]).lower()


def test_duplicates_versions_and_incremental_reindex(client):
    new_patient("MRN-TEST-VER")
    first = upload(client, "MRN-TEST-VER", b"Clinic note.\n\nPlan: start lisinopril 10 mg.", name="v1.txt").json()
    run_pending_jobs()
    doc_id = first["document"]["id"]

    dup = upload(client, "MRN-TEST-VER", b"Clinic note.\n\nPlan: start lisinopril 10 mg.", name="copy.txt")
    assert dup.status_code == 200 and dup.json()["status"] == "duplicate" and dup.json()["document"]["id"] == doc_id and dup.json()["job_id"] is None

    v2 = client.post(f"/api/documents/{doc_id}/versions", files={"file": ("v2.txt", b"Clinic note.\n\nPlan: lisinopril stopped, losartan 50 mg started.", "text/plain")},
                     headers=auth(client))
    assert v2.status_code == 202 and v2.json()["status"] == "new_version"
    before = client.get(f"/api/documents/{doc_id}/chunks", headers=auth(client)).json()
    assert before and all("losartan" not in c["text"] for c in before)  # v1 stays searchable while v2 is queued
    run_pending_jobs()
    detail = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()
    after = client.get(f"/api/documents/{doc_id}/chunks", headers=auth(client)).json()
    assert detail["current_version"] == 2 and [v["version"] for v in detail["versions"]] == [1, 2]
    assert after and all("losartan" in c["text"] or "Clinic" in c["text"] for c in after) and not any("start lisinopril" in c["text"] for c in after)
    same_again = client.post(f"/api/documents/{doc_id}/versions", files={"file": ("v2.txt", b"Clinic note.\n\nPlan: lisinopril stopped, losartan 50 mg started.", "text/plain")},
                             headers=auth(client))
    assert same_again.json()["status"] == "duplicate"


def test_reindex_touches_only_stale_versions(client, monkeypatch):
    from backend.app.config import settings

    h = auth(client, "rivera")
    assert client.get("/api/ingestion/stats", headers=h).json()["stale_versions"] == 0
    monkeypatch.setattr(settings, "chunk_target_chars", 400)  # chunker config changed -> everything indexed is stale
    stale = client.get("/api/ingestion/stats", headers=h).json()["stale_versions"]
    assert stale > 0
    r = client.post("/api/ingestion/reindex", json={"scope": "stale"}, headers=h)
    assert r.json()["jobs_enqueued"] == stale
    run_pending_jobs()
    assert client.get("/api/ingestion/stats", headers=h).json()["stale_versions"] == 0
    assert client.post("/api/ingestion/reindex", json={"scope": "stale"}, headers=h).json()["jobs_enqueued"] == 0


def test_delete_removes_chunks_immediately_then_purges_originals(client):
    from backend.app.database import SessionLocal
    from backend.app.ingestion.storage import get_storage
    from backend.app.models import DocumentChunk, DocumentVersion

    new_patient("MRN-TEST-DEL")
    doc_id = upload(client, "MRN-TEST-DEL", b"ER note.\n\nFall at home, hit head, on apixaban.", name="er.txt", doc_type="er_note").json()["document"]["id"]
    run_pending_jobs()
    db = SessionLocal()
    key = db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).one().storage_key
    db.close()
    assert get_storage().exists(key)

    assert client.delete(f"/api/documents/{doc_id}", headers=auth(client, "lee")).status_code == 403  # triage nurses cannot delete records
    r = client.delete(f"/api/documents/{doc_id}", headers=auth(client, "patel"))
    assert r.status_code == 202 and r.json()["kind"] == "DELETE"
    db = SessionLocal()
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).count() == 0  # gone before the job runs
    db.close()
    assert client.get("/api/patients/MRN-TEST-DEL", headers=auth(client)).json()["has_history"] is False
    assert all(d["id"] != doc_id for d in client.get("/api/patients/MRN-TEST-DEL/documents", headers=auth(client)).json())

    run_pending_jobs()
    assert not get_storage().exists(key)
    detail = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()
    assert detail["status"] == "DELETED" and detail["versions"][0]["original_available"] is False  # tombstone, no content
    assert client.delete(f"/api/documents/{doc_id}", headers=auth(client, "patel")).status_code == 409


def test_document_permissions(client):
    new_patient("MRN-TEST-DOCS")
    assert upload(client, "MRN-TEST-DOCS", b"note text here", user="viewer").status_code == 403
    assert upload(client, "MRN-TEST-DOCS", b"note text here", user="admin").status_code == 403  # admins have no PHI write access
    assert client.get("/api/patients/MRN-TEST-DOCS/documents", headers=auth(client, "admin")).status_code == 403
    assert client.get("/api/ingestion/jobs", headers=auth(client, "lee")).status_code == 403
    assert client.get("/api/ingestion/jobs", headers=auth(client, "rivera")).status_code == 200
    assert upload(client, "MRN-NOPE", b"note text here").status_code == 404


def test_image_only_pdf_is_held_for_review_or_fails_permanently_without_ocr(client, monkeypatch):
    """Phase 6: a scanned PDF goes to OCR; with no readable text it is held for human review (never indexed blind).
    With OCR unavailable it fails permanently on the first attempt, with a clear message."""
    from backend.app.config import settings
    from backend.app.intake import ocr

    new_patient("MRN-TEST-DOCS")
    blank = fitz.open()
    blank.new_page()
    if ocr.get_ocr_engine() is not None:
        doc_id = upload(client, "MRN-TEST-DOCS", blank.tobytes(), name="scan.pdf", doc_type="imaging_report").json()["document"]["id"]
        run_pending_jobs()
        d = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()
        assert d["status"] == "NEEDS_REVIEW" and d["chunk_count"] == 0

    monkeypatch.setattr(settings, "ocr_engine", "none")
    ocr.get_ocr_engine.cache_clear()
    try:
        blank2 = fitz.open()
        blank2.new_page().insert_text((72, 72), " ")  # different bytes, still no extractable text
        doc_id = upload(client, "MRN-TEST-DOCS", blank2.tobytes(), name="scan2.pdf", doc_type="imaging_report").json()["document"]["id"]
        run_pending_jobs()
        d = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()
        assert d["status"] == "FAILED" and d["jobs"][-1]["status"] == "FAILED" and d["jobs"][-1]["attempts"] == 1
        assert "OCR is not available" in d["latest_job_error"]
    finally:
        ocr.get_ocr_engine.cache_clear()


def test_transient_failures_back_off_retry_then_dead_letter(client, monkeypatch):
    new_patient("MRN-TEST-RETRY")
    calls = {"n": 0}
    real = service.get_embedder

    def flaky(*a, **k):
        calls["n"] += 1
        raise RuntimeError("embedding service timeout")

    monkeypatch.setattr(service, "get_embedder", flaky)
    doc_id = upload(client, "MRN-TEST-RETRY", b"Clinic note.\n\nStable asthma.", name="a.txt").json()["document"]["id"]
    now = utc_now()
    run_pending_jobs(now=now)
    job = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()["jobs"][-1]
    assert job["status"] == "QUEUED" and job["attempts"] == 1 and "timeout" in job["last_error"]
    assert run_pending_jobs(now=now) == 0                      # not due yet (backoff)
    run_pending_jobs(now=now + timedelta(hours=1))
    run_pending_jobs(now=now + timedelta(hours=2))
    job = client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()["jobs"][-1]
    assert job["status"] == "DEAD" and job["attempts"] == 3 and calls["n"] == 3
    assert client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()["status"] == "FAILED"
    dead = client.get("/api/ingestion/jobs?status=DEAD", headers=auth(client, "rivera")).json()
    assert job["id"] in {j["id"] for j in dead}

    monkeypatch.setattr(service, "get_embedder", real)   # the dependency recovers
    assert client.post(f"/api/ingestion/jobs/{job['id']}/retry", headers=auth(client, "rivera")).json()["status"] == "QUEUED"
    run_pending_jobs()
    assert client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()["status"] == "READY"


def test_crashed_worker_lease_is_recovered(client):
    from backend.app.database import SessionLocal
    from backend.app.models import IngestionJob

    new_patient("MRN-TEST-LEASE")
    doc_id = upload(client, "MRN-TEST-LEASE", b"Clinic note.\n\nKnee pain.", name="k.txt").json()["document"]["id"]
    db = SessionLocal()
    job = db.query(IngestionJob).filter(IngestionJob.document_id == doc_id).one()
    job.status, job.started_at, job.attempts = "RUNNING", utc_now() - timedelta(hours=1), 1  # a worker died mid-job
    db.commit()
    db.close()
    run_pending_jobs()
    assert client.get(f"/api/documents/{doc_id}", headers=auth(client)).json()["status"] == "READY"


def test_seeded_history_goes_through_the_same_pipeline(client):
    docs = client.get("/api/patients/MRN-A3F9/documents", headers=auth(client)).json()
    assert docs and all(d["status"] == "READY" and d["chunk_count"] >= 1 and d["is_synthetic"] for d in docs)
    stats = client.get("/api/ingestion/stats", headers=auth(client, "admin")).json()
    assert stats["vector_backend"] == "sql" and stats["embedding_model"] == "hash-v2" and stats["chunks"] > 100


# ------------------------------------------------------------------ migrations


def test_migrations_build_exactly_the_models_and_round_trip(tmp_path):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine, inspect, text

    from backend.app.database import Base
    from backend.app.db_migrate import alembic_config, run_migrations

    eng = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    run_migrations(eng)
    with eng.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []

    cfg = alembic_config()
    with eng.begin() as conn:
        cfg.attributes["connection"] = conn
        command.downgrade(cfg, "0001")
        assert "document_chunks" not in inspect(conn).get_table_names()
        command.upgrade(cfg, "head")
        assert "document_chunks" in inspect(conn).get_table_names()

    # A database built by the pre-Alembic code (baseline tables, no alembic_version) is stamped then upgraded.
    old = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with old.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "0001")
        conn.execute(text("DROP TABLE alembic_version"))
        conn.execute(text("INSERT INTO patients (id, mrn, age_years, gender, is_anonymized, created_at) VALUES ('p1', 'MRN-OLD', 50, 'M', 0, '2026-01-01 00:00:00')"))
    run_migrations(old)
    with old.connect() as conn:
        assert conn.execute(text("SELECT mrn FROM patients")).scalar() == "MRN-OLD"  # data survived
        assert "ingestion_jobs" in inspect(conn).get_table_names()


# ------------------------------------------------------------------ optional Chroma backend


def test_chroma_backend_is_patient_filtered_and_persistent(client, tmp_path):
    pytest.importorskip("chromadb")
    from backend.app.database import SessionLocal
    from backend.app.ingestion.embeddings import get_embedder
    from backend.app.ingestion.vector_index import ChromaVectorIndex
    from backend.app.models import DocumentChunk, Patient

    db = SessionLocal()
    a = db.query(Patient).filter(Patient.mrn == "MRN-A3F9").one()
    b = db.query(Patient).filter(Patient.mrn == "MRN-GER-005").one()
    rows = db.query(DocumentChunk).filter(DocumentChunk.patient_id.in_([a.id, b.id])).all()
    ChromaVectorIndex(str(tmp_path)).upsert(db, rows)

    reopened = ChromaVectorIndex(str(tmp_path))  # new client on the same folder: data persisted
    q = get_embedder().embed(["anticoagulant fall"])[0]
    hits = reopened.query(db, b.id, q, 5, "hash-v2")
    owner = {c.id: c.patient_id for c in rows}
    assert hits and all(owner[h.chunk_id] == b.id for h in hits)
    some_doc = hits[0].document_id
    reopened.delete_document(db, some_doc)
    assert all(h.document_id != some_doc for h in reopened.query(db, b.id, q, 20, "hash-v2"))
    db.close()
