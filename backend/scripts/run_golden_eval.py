"""
Opt-in golden eval WITH the LLM (uses OPENAI_API_KEY from backend/.env; costs tokens).
CI runs the deterministic checks only (backend/tests/test_golden.py).

  python -m backend.scripts.run_golden_eval
"""
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Throwaway DB so the eval never touches real data.
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp(), 'eval.db').as_posix()}"
os.environ["SEED_ON_STARTUP"] = "false"
os.environ["DETERIORATION_MONITOR_ENABLED"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.config import settings  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.scripts.seed_data import seed  # noqa: E402

GOLDEN = PROJECT_ROOT / "backend" / "tests" / "golden" / "scenarios.json"


def main() -> None:
    if not settings.llm_configured:
        raise SystemExit("No LLM configured (OPENAI_API_KEY / LLM_ENABLED).")
    seed(reset=True)
    scenarios = json.loads(GOLDEN.read_text(encoding="utf-8"))["scenarios"]
    exact = within1 = under = 0
    with TestClient(app) as client:
        for s in scenarios:
            body = client.post("/api/triage", json={
                "request_id": str(uuid.uuid4()), "mrn": s["mrn"], "age_years": s["age_years"], "gender": s["gender"],
                "chief_complaint": s["complaint"], "vitals": s["vitals"],
            }).json()
            got, exp = body["recommended_esi"], s["expected_esi"]
            exact += got == exp
            within1 += abs(got - exp) <= 1
            under += got > exp
            c = body["components"]
            print(f"{s['id']:<26} expected {exp} got {got} (llm {c['llm_esi']}, ml {c['ml_esi']}, floor {c['rule_floor_esi']}) degraded={body['degraded']}")
    n = len(scenarios)
    print(f"\nexact {exact}/{n}  within-1 {within1}/{n}  under-triaged {under}/{n}  model={settings.llm_model or 'default'}")


if __name__ == "__main__":
    main()
