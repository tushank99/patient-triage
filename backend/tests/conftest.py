"""
Test setup: isolated SQLite DB, LLM disabled (no network), no background monitor.
Environment variables must be set before any backend module is imported.
"""
import json
import os
import tempfile
import uuid
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="triage-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["ENVIRONMENT"] = "test"
os.environ["OPENAI_API_KEY"] = ""
os.environ["LLM_ENABLED"] = "false"
os.environ["SEED_ON_STARTUP"] = "false"
os.environ["DETERIORATION_MONITOR_ENABLED"] = "false"
os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret-123456"
os.environ["DEMO_USER_PASSWORD"] = "test-password-123456"
os.environ["STORAGE_PATH"] = str(Path(_TMP, "uploads"))
os.environ["CHROMA_PATH"] = str(Path(_TMP, "chroma"))
os.environ["INGESTION_WORKER_ENABLED"] = "false"  # tests drive jobs explicitly with run_pending_jobs()
os.environ["EMBEDDING_MODEL"] = "hash-v2"
os.environ["VECTOR_BACKEND"] = "sql"
os.environ["RATE_LIMIT_ENABLED"] = "false"  # test_production enables it explicitly
os.environ["REDIS_URL"] = ""
os.environ["JOB_BACKEND"] = "db"
os.environ["BACKUP_DIR"] = str(Path(_TMP, "backups"))
if os.environ.get("TEST_DATABASE_URL"):  # CI Postgres job
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "backend" / "tests" / "golden" / "scenarios.json"
PASSWORD = "test-password-123456"
_header_cache: dict = {}


@pytest.fixture(scope="session")
def seeded():
    from backend.scripts.seed_data import seed

    return seed(reset=True, users=True)


@pytest.fixture(scope="session")
def client(seeded):
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def auth(client: TestClient, username: str = "lee") -> dict:
    """Authorization header for a seeded demo user (cached)."""
    if username not in _header_cache:
        r = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
        assert r.status_code == 200, r.text
        _header_cache[username] = {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _header_cache[username]


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    from backend.app.main import app

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def triage_payload(scenario: dict, **overrides) -> dict:
    body = {
        "request_id": str(uuid.uuid4()),
        "mrn": scenario.get("mrn"),
        "age_years": scenario["age_years"],
        "gender": scenario["gender"],
        "chief_complaint": scenario["complaint"],
        "vitals": scenario["vitals"],
    }
    body.update(overrides)
    return body


def fake_llm(esi=4, category="RESPIRATORY", cited=(), risk_factors=(), captured=None):
    """A stand-in for assess_with_llm (the pipeline's LLMFn)."""
    from backend.app.llm.assessor import LLMAssessment, LLMCallMeta, LLMInput, LLMOutcome

    def fn(inp: LLMInput) -> LLMOutcome:
        if captured is not None:
            captured.append(inp)
        return LLMOutcome(
            LLMAssessment(category=category, esi=esi, rationale="Looks well.", risk_factors=list(risk_factors), cited_document_ids=list(cited)),
            LLMCallMeta("fake", "fake-model", "triage_assessment.v3", 1, 5, None, None, False, "sha", "request-text"),
        )

    return fn
