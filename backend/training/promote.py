"""
Promote a candidate run to the shipped model, but only through the gate.

  python -m backend.training.promote --run <run_id> --channel demo
  python -m backend.training.promote --run <run_id> --channel production --signoff "Dr A. Patel, ED physician"
  python -m backend.training.promote --run <run_id> --channel research --dry-run

Copies the bundle to backend/app/ml/artifacts/ (server) and frontend/public/models/ (browser), so both are
produced from the same fitted model in the same step. Exit code 1 if the gate fails.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.clinical.rules_engine import RULES_PATH  # noqa: E402
from backend.app.ml.model import ARTIFACT_DIR, CARD_PATH, TriageModel  # noqa: E402
from backend.training.gate import evaluate_gate  # noqa: E402
from backend.training.golden import golden_regression  # noqa: E402

RUNS_DIR = ROOT / "backend" / "ml_runs"
FRONTEND_DIR = ROOT / "frontend" / "public" / "models"


def promote(run_id: str, channel: str, signoff: str | None = None, dry_run: bool = False, runs_dir: Path = RUNS_DIR,
            artifact_dir: Path | None = None, frontend_dir: Path | None = None, card_path: Path | None = None) -> dict:
    import joblib

    artifact_dir, frontend_dir = Path(artifact_dir or ARTIFACT_DIR), Path(frontend_dir or FRONTEND_DIR)
    card_path = Path(card_path or CARD_PATH)
    run = runs_dir / run_id
    card = json.loads((run / "model_card.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "triage_model.meta.json").read_text(encoding="utf-8"))
    shipped = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else None

    candidate = TriageModel(joblib.load(run / "triage_model.joblib"), meta)
    result = evaluate_gate(card, shipped, channel, signoff, golden_regression(candidate))
    print(result.table())
    if not result.passed:
        return {"passed": False, "gate": result.as_dict()}
    if dry_run:
        print("Dry run: nothing copied.")
        return {"passed": True, "gate": result.as_dict(), "copied": False}

    artifact_dir.mkdir(parents=True, exist_ok=True)
    frontend_dir.mkdir(parents=True, exist_ok=True)
    for name in ("triage_model.joblib", "triage_model.onnx", "triage_model.meta.json"):
        shutil.copyfile(run / name, artifact_dir / name)
    for name in ("triage_model.onnx", "triage_model.meta.json", "parity_vectors.json"):
        shutil.copyfile(run / name, frontend_dir / name)
    shutil.copyfile(RULES_PATH, frontend_dir / "safety_rules.json")  # browser applies the same deterministic rules
    card["promoted"] = {"channel": channel, "at": datetime.now(timezone.utc).isoformat(), "signoff": signoff, "gate": result.as_dict()}
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    shutil.copyfile(run / "report.md", artifact_dir / "model_report.md")
    print(f"Promoted {card['model_version']} ({card['label_source']}) to the {channel!r} channel.")
    return {"passed": True, "gate": result.as_dict(), "copied": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--channel", choices=["demo", "research", "production"], required=True)
    ap.add_argument("--signoff")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if promote(a.run, a.channel, a.signoff, a.dry_run)["passed"] else 1)


if __name__ == "__main__":
    main()
