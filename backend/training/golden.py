"""
Golden clinical scenarios, scored with a CANDIDATE model + the safety rules (no LLM, no DB).
Dataset-independent, so it catches regressions even when the training data changes between versions.
"""
import json
from pathlib import Path
from typing import Dict

from backend.app.clinical.rules_engine import evaluate_rules, rule_floor
from backend.app.ml.model import TriageModel
from backend.app.services.history_store import derive_facts

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "backend" / "tests" / "golden" / "scenarios.json"
MASTER = ROOT / "backend" / "data" / "patient_master.json"


def golden_regression(model: TriageModel) -> Dict:
    scenarios = json.loads(GOLDEN.read_text(encoding="utf-8"))
    master = {p["mrn"]: p for p in json.loads(MASTER.read_text(encoding="utf-8"))["patients"]}
    hits = under = 0
    for s in scenarios["scenarios"]:
        patient = master.get(s["mrn"]) if s["mrn"] else None
        has_history = bool(patient and patient["profile"] != "none")
        facts = derive_facts({k: patient.get(k, []) for k in ("conditions", "medications", "allergies")}) if has_history else {}
        values = {"age_years": s["age_years"], **s["vitals"], "has_prior_history": has_history}
        ml = model.predict(values, explain=False)
        floor = rule_floor(evaluate_rules({**values, "complaint": s["complaint"], "history": facts}))
        final = min(ml.esi, floor) if floor is not None else ml.esi
        hits += final == s["expected_esi"]
        if s["deterministic_max_esi"] is not None and final > s["deterministic_max_esi"]:
            under += 1
    n = len(scenarios["scenarios"])
    return {"n": n, "hits": hits, "exact_match": hits / n, "under_triaged": under,
            "baseline": scenarios["regression_baseline"]["deterministic_exact_match_min"]}
