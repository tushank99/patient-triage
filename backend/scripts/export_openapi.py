"""
Write the API contract snapshot used by the contract test and the frontend type generator.

  python -m backend.scripts.export_openapi
  cd frontend && npm run gen:api
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SEED_ON_STARTUP", "false")

from backend.app.main import app  # noqa: E402

OUT = PROJECT_ROOT / "contracts" / "openapi.json"

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
