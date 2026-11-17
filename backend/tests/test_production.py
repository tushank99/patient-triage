"""Phase 7: rate limiting, versioning, observability, security, encryption, backups, retention, Celery dispatch,
concurrency and failure injection (DB, Redis, embedder, storage, LLM outages)."""
import asyncio
import json
import logging
import uuid

import httpx
import pytest
from sqlalchemy.exc import OperationalError

from backend.app.config import settings
from backend.app.database import SessionLocal, get_db
from backend.app.deps import get_llm_fn
from backend.app.ingestion.worker import run_pending_jobs
from backend.app.llm.assessor import LLMUnavailable
from backend.app.main import app
from backend.tests.conftest import auth, fake_llm
from backend.tests.test_ingestion import new_patient, upload
from backend.tests.test_pipeline_safety import SILENT_HYPOXIA


def payload(**kw):
    s = SILENT_HYPOXIA
    body = {"request_id": str(uuid.uuid4()), "mrn": s["mrn"], "age_years": s["age_years"], "gender": s["gender"],
            "chief_complaint": s["complaint"], "vitals": s["vitals"]}
    body.update(kw)
    return body


@pytest.fixture
def limits(monkeypatch):
    from backend.app.ops import ratelimit

    ratelimit.reset_for_tests()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    yield monkeypatch
    ratelimit.reset_for_tests()


# ------------------------------------------------------------------ versioning


def test_v1_and_legacy_prefix_both_work_and_legacy_is_marked_deprecated(client):
    h = auth(client)
    v1 = client.get("/api/v1/auth/me", headers=h)
    old = client.get("/api/auth/me", headers=h)
    assert v1.status_code == old.status_code == 200 and v1.json() == old.json()
    assert "Deprecation" not in v1.headers
    assert old.headers["Deprecation"] == "true" and "/api/v1/auth/me" in old.headers["Link"] and old.headers["Sunset"]
    paths = app.openapi()["paths"]
    assert all(p.startswith("/api/v1/") for p in paths), "legacy aliases must stay out of the contract"


# ------------------------------------------------------------------ rate limiting


def test_rate_limit_returns_429_envelope_with_retry_after(client, limits):
    limits.setitem(settings.rate_limits, "login", "3/60")
    codes = [client.post("/api/v1/auth/login", json={"username": "nobody", "password": "wrong-password"}).status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]
    r = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "wrong-password"})
    assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) > 0
    assert r.headers["X-RateLimit-Limit"] == "3"


def test_rate_limit_is_per_user_and_health_is_exempt(client, limits):
    limits.setitem(settings.rate_limits, "default", "2/60")
    for _ in range(4):
        assert client.get("/healthz").status_code == 200
    assert [client.get("/api/v1/auth/me", headers=auth(client, "lee")).status_code for _ in range(3)] == [200, 200, 429]
    assert client.get("/api/v1/auth/me", headers=auth(client, "patel")).status_code == 200  # separate bucket


def test_rate_limit_uses_redis_and_falls_back_to_memory_when_redis_dies(client, limits):
    fakeredis = pytest.importorskip("fakeredis")
    from backend.app.ops import ratelimit

    server = fakeredis.FakeServer()
    limits.setattr(settings, "redis_url", "redis://fake")
    limits.setattr(ratelimit, "_redis", fakeredis.FakeRedis(server=server))
    assert ratelimit.hit("t:x", 60)[2] == "redis"
    assert any(k.startswith(b"rl:t:x:") for k in ratelimit._redis.keys())

    server.connected = False  # Redis outage: requests still succeed, limits continue in memory
    n, _, backend = ratelimit.hit("t:x", 60)
    assert backend == "memory" and n == 1
    assert client.get("/api/v1/auth/me", headers=auth(client)).status_code == 200


# ------------------------------------------------------------------ observability + security


def test_metrics_endpoint_exposes_latency_stage_and_cost_metrics(client, monkeypatch):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=2)
    assert client.post("/api/v1/triage", json=payload(), headers=auth(client)).status_code == 200
    text = client.get("/metrics").text
    assert 'http_request_duration_seconds_count{method="POST",route="/api/v1/triage"' in text
    for stage in ("retrieval", "ml", "llm", "rules", "grounding"):
        assert f'pipeline_stage_seconds_count{{stage="{stage}"}}' in text
    assert "ingestion_oldest_queued_seconds" in text and "assessments_total" in text

    monkeypatch.setattr(settings, "metrics_token", "s3cret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_llm_cost_is_estimated_from_token_counts():
    from backend.app.ops.observability import LLM_COST, llm_cost, record_llm_call

    assert llm_cost("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert llm_cost("unknown-model", 1000, 1000) is None
    before = LLM_COST.labels("gpt-4o-mini")._value.get()
    record_llm_call("openai", "gpt-4o-mini", True, 2000, 0)
    assert LLM_COST.labels("gpt-4o-mini")._value.get() == pytest.approx(before + 0.0003)


def test_health_readiness_and_security_headers(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"]["database"] == "ok"
    r = client.get("/api/v1/auth/me", headers=auth(client))
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["Cache-Control"] == "no-store"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_log_filter_scrubs_identifiers():
    from backend.app.ops.security import PiiLogFilter

    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "lookup %s by a@b.org phone 415-555-0100 dob 1950-02-03", ("MRN-12345",), None)
    PiiLogFilter().filter(rec)
    msg = rec.getMessage()
    assert "12345" not in msg and "a@b.org" not in msg and "555-0100" not in msg and "1950-02-03" not in msg


def test_production_checks_refuse_unsafe_config(monkeypatch):
    from backend.app.ops.security import production_checks

    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "database_url", "postgresql://u:p@host/db")
    monkeypatch.setattr(settings, "encryption_key", "")
    problems = " ".join(production_checks())
    assert "TLS" in problems and "ENCRYPTION_KEY" in problems


# ------------------------------------------------------------------ encryption, backups, retention


@pytest.fixture
def fernet_key(monkeypatch):
    from cryptography.fernet import Fernet

    from backend.app.ingestion.storage import get_storage

    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


def test_originals_are_encrypted_at_rest_and_still_readable(client, fernet_key):
    from backend.app.ingestion.storage import _raw_storage, get_storage
    from backend.app.models import DocumentVersion
    from backend.app.ops.security import ENC_MAGIC

    new_patient("MRN-TEST-ENC")
    doc_id = upload(client, "MRN-TEST-ENC", b"Clinic note.\n\nKnown asthma, uses salbutamol.", name="enc.txt").json()["document"]["id"]
    run_pending_jobs()
    db = SessionLocal()
    key = db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).one().storage_key
    db.close()
    raw = _raw_storage().get(key)
    assert raw.startswith(ENC_MAGIC) and b"salbutamol" not in raw
    assert b"salbutamol" in get_storage().get(key)


def test_backup_is_encrypted_rotated_and_restore_drill_passes(client, fernet_key, monkeypatch, tmp_path):
    from backend.app.ops import backup

    monkeypatch.setattr(settings, "backup_keep", 2)
    for _ in range(3):
        m = backup.create_backup(out_dir=str(tmp_path))
    assert len(backup.list_backups(str(tmp_path))) == 2 and m["encrypted"]
    path = tmp_path / m["file"]
    assert not path.read_bytes().startswith(b"SQLite format")
    result = backup.verify(str(path))
    assert result["ok"], result
    assert result["restored_counts"]["patients"] > 0


def test_deletion_reaches_vector_index_storage_and_caches(client, monkeypatch):
    """A delete whose purge half-failed is completed by retention: the external index is always cleaned."""
    from backend.app.ingestion import vector_index
    from backend.app.ingestion.storage import get_storage
    from backend.app.models import ClinicalDocument, DocumentChunk, DocumentVersion
    from backend.app.ops.retention import run_retention

    new_patient("MRN-TEST-RET")
    doc_id = upload(client, "MRN-TEST-RET", b"ER note.\n\nChest pain, troponin negative.", name="ret.txt", doc_type="er_note").json()["document"]["id"]
    run_pending_jobs()
    db = SessionLocal()
    key = db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).one().storage_key
    db.close()

    deleted_from_index = []

    class SpyIndex(vector_index.SqlVectorIndex):
        def delete_document(self, db, document_id):
            deleted_from_index.append(document_id)
            super().delete_document(db, document_id)

    # Simulate a crash between "mark deleted" and the purge job: status DELETED, original still in storage.
    db = SessionLocal()
    db.get(ClinicalDocument, doc_id).status = "DELETED"
    db.commit()
    db.close()
    import backend.app.ingestion.service as service

    spy = SpyIndex()
    monkeypatch.setattr(service, "get_index", lambda name=None: spy)
    monkeypatch.setattr(vector_index, "get_index", lambda name=None: spy)
    stats = run_retention()
    assert stats["deleted_documents_purged"] >= 1 and doc_id in deleted_from_index
    assert not get_storage().exists(key)
    db = SessionLocal()
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).count() == 0
    db.close()


def test_deletion_reaches_chroma(client, monkeypatch):
    pytest.importorskip("chromadb")
    import numpy as np

    from backend.app.ingestion.vector_index import ChromaVectorIndex
    from backend.app.models import DocumentChunk

    new_patient("MRN-TEST-CHR")
    doc_id = upload(client, "MRN-TEST-CHR", b"Clinic note.\n\nDiabetes on metformin.", name="c.txt").json()["document"]["id"]
    run_pending_jobs()
    idx = ChromaVectorIndex(settings.chroma_path)
    db = SessionLocal()
    rows = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).all()
    idx.upsert(db, rows)
    pid, model = rows[0].patient_id, rows[0].embedding_model
    vec = np.frombuffer(rows[0].embedding, dtype=np.float32) if isinstance(rows[0].embedding, bytes) else np.asarray(rows[0].embedding)
    col = idx._collection(model)
    assert col.get(where={"document_id": doc_id})["ids"]
    idx.delete_document(db, doc_id)
    assert not col.get(where={"document_id": doc_id})["ids"]
    db.close()
    assert pid and vec.size


def test_retention_prunes_old_access_logs_and_finished_jobs(client):
    from datetime import timedelta

    from backend.app.models import AccessLog, IngestionJob, utc_now
    from backend.app.ops.retention import run_retention

    db = SessionLocal()
    old = utc_now() - timedelta(days=settings.retention_access_log_days + 5)
    db.add(AccessLog(username="ghost", action="VISIT_READ", timestamp=old))
    db.add(IngestionJob(kind="INGEST", document_id="gone", status="SUCCEEDED", created_by="test", finished_at=utc_now() - timedelta(days=settings.retention_job_days + 1)))
    db.commit()
    db.close()
    stats = run_retention()
    assert stats["access_logs_deleted"] >= 1 and stats["jobs_deleted"] >= 1


# ------------------------------------------------------------------ Celery


def test_celery_dispatch_happens_only_after_commit(monkeypatch):
    from backend.app.ingestion.service import enqueue
    from backend.app.ops import celery_app

    sent = []
    monkeypatch.setattr(settings, "job_backend", "celery")
    monkeypatch.setattr(celery_app, "dispatch", lambda job_id, heavy=False: sent.append(job_id) or True)

    db = SessionLocal()
    job = enqueue(db, "REINDEX", "doc-x", None, "test")
    assert sent == []  # not before commit
    db.rollback()
    assert sent == []  # never for a rolled-back transaction
    job = enqueue(db, "REINDEX", "doc-y", None, "test")
    db.commit()
    assert sent == [job.id]
    db.delete(job)
    db.commit()
    db.close()


def test_celery_task_claims_by_id_and_ignores_duplicate_messages(client):
    from backend.app.ops import celery_app

    new_patient("MRN-TEST-CEL")
    doc_id = upload(client, "MRN-TEST-CEL", b"Clinic note.\n\nHypertension on amlodipine.", name="cel.txt").json()["document"]["id"]
    from backend.app.models import IngestionJob

    db = SessionLocal()
    job_id = db.query(IngestionJob.id).filter(IngestionJob.document_id == doc_id).scalar()
    db.close()
    celery_app.celery.conf.task_always_eager = True
    try:
        assert celery_app.run_ingestion_job.delay(job_id).get() is True
        assert celery_app.run_ingestion_job.delay(job_id).get() is False  # redelivered: already done, no rerun
    finally:
        celery_app.celery.conf.task_always_eager = False
    db = SessionLocal()
    assert db.get(IngestionJob, job_id).status == "SUCCEEDED" and db.get(IngestionJob, job_id).attempts == 1
    db.close()


# ------------------------------------------------------------------ concurrency


def test_concurrent_triage_requests_and_idempotent_retries(client):
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=3)
    headers = auth(client)
    same = payload()

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            distinct = [c.post("/api/v1/triage", json=payload(mrn=None), headers=headers) for _ in range(8)]
            dupes = [c.post("/api/v1/triage", json=same, headers=headers) for _ in range(4)]
            return await asyncio.gather(*distinct, *dupes)

    rs = asyncio.run(run())
    assert all(r.status_code == 200 for r in rs), [r.text for r in rs if r.status_code != 200][:1]
    dupe_ids = {r.json()["visit_id"] for r in rs[8:]}
    assert len(dupe_ids) == 1  # same request_id -> one visit, however the retries interleave
    assert len({r.json()["visit_id"] for r in rs[:8]}) == 8


def test_concurrent_workers_never_run_a_job_twice(client):
    from concurrent.futures import ThreadPoolExecutor

    from backend.app.ingestion.worker import run_one
    from backend.app.models import IngestionJob

    new_patient("MRN-TEST-CON")
    ids = [upload(client, "MRN-TEST-CON", f"Note {i}.\n\nStable, follow up {i}.".encode(), name=f"n{i}.txt").json()["document"]["id"] for i in range(4)]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda _: [run_one() for _ in range(3)], range(6)))
    db = SessionLocal()
    jobs = db.query(IngestionJob).filter(IngestionJob.document_id.in_(ids)).all()
    assert all(j.status == "SUCCEEDED" and j.attempts == 1 for j in jobs), [(j.status, j.attempts) for j in jobs]
    db.close()


# ------------------------------------------------------------------ failure injection


def test_database_outage_returns_503_envelope(client):
    def broken_db():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app.dependency_overrides[get_db] = broken_db
    r = client.get("/api/v1/queue", headers=auth(client))
    assert r.status_code == 503 and r.json()["error"]["code"] == "database_unavailable" and r.headers["Retry-After"]


def test_embedder_outage_falls_back_to_keyword_retrieval(client, monkeypatch):
    import backend.app.retrieval.retriever as retriever

    new_patient("MRN-TEST-EMB")
    upload(client, "MRN-TEST-EMB", b"Clinic note.\n\nCOPD, home oxygen 2L.", name="copd.txt")
    run_pending_jobs()

    def down(*a, **k):
        raise ConnectionError("embedding service down")

    monkeypatch.setattr(retriever, "retrieve", down)
    app.dependency_overrides[get_llm_fn] = lambda: fake_llm(esi=3)
    r = client.post("/api/v1/triage", json=payload(mrn="MRN-TEST-EMB", chief_complaint="short of breath, COPD"), headers=auth(client))
    body = r.json()
    assert r.status_code == 200 and "retrieval_degraded" in body["degraded"]
    assert body["history"]["retrieval_method"] == "keyword" and body["recommended_esi"] <= 2


def test_llm_and_ml_outage_still_produce_a_rules_floor(client, monkeypatch):
    import backend.app.services.triage_pipeline as pipeline

    def broken(_):
        raise LLMUnavailable("llm_unavailable")

    app.dependency_overrides[get_llm_fn] = lambda: broken
    monkeypatch.setattr(pipeline, "get_model", lambda: None)
    body = client.post("/api/v1/triage", json=payload(), headers=auth(client)).json()
    assert {"llm_unavailable", "ml_unavailable"} <= set(body["degraded"])
    assert body["recommended_esi"] <= 2 and body["requires_clinician_review"]


def test_storage_outage_is_retried_not_lost(client, monkeypatch):
    from backend.app.ingestion import service
    from backend.app.models import IngestionJob

    new_patient("MRN-TEST-STO")
    doc_id = upload(client, "MRN-TEST-STO", b"Clinic note.\n\nGout, on allopurinol.", name="s.txt").json()["document"]["id"]

    class Down:
        name = "down"

        def get(self, key):
            raise ConnectionError("S3 unreachable")

    monkeypatch.setattr(service, "get_storage", lambda: Down())
    run_pending_jobs()
    db = SessionLocal()
    job = db.query(IngestionJob).filter(IngestionJob.document_id == doc_id).one()
    assert job.status == "QUEUED" and job.attempts == 1 and "unreachable" in (job.last_error or "")  # backoff, not FAILED
    db.close()
