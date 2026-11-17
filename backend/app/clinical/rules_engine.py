"""
Deterministic safety floor.

Rules live in safety_rules.json (data, not code) so the browser fallback model can evaluate
the exact same rules (frontend/src/lib/onnx-engine.ts). The floor is the most acute ESI of
all fired rules; the pipeline takes min(model_esi, floor), so rules can only escalate.
"""
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional

RULES_PATH = os.path.join(os.path.dirname(__file__), "safety_rules.json")


@dataclass(frozen=True)
class FiredRule:
    id: str
    version: int
    esi: int
    reason: str
    source: str


@lru_cache(maxsize=1)
def load_ruleset() -> Dict[str, Any]:
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def ruleset_version() -> str:
    return load_ruleset()["ruleset_version"]


def _resolve(inputs: Mapping[str, Any], field: str) -> Any:
    cur: Any = inputs
    for part in field.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _condition_holds(inputs: Mapping[str, Any], cond: Mapping[str, Any]) -> bool:
    actual = _resolve(inputs, cond["field"])
    if actual is None:
        return False  # never fire on data that was not collected
    op, expected = cond["op"], cond["value"]
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "==":
        return actual == expected
    text = str(actual).lower()
    if op == "contains_any":
        return any(phrase.lower() in text for phrase in expected)
    if op == "has_word_any":
        return any(re.search(r"\b" + re.escape(w.lower()) + r"\b", text) for w in expected)
    raise ValueError(f"Unknown rule operator: {op}")


def evaluate_rules(inputs: Mapping[str, Any]) -> List[FiredRule]:
    """inputs: feature-schema field names + 'complaint' + optional 'history' dict."""
    fired: List[FiredRule] = []
    for rule in load_ruleset()["rules"]:
        if not all(_condition_holds(inputs, c) for c in rule.get("all", [])):
            continue
        any_conds = rule.get("any")
        if any_conds and not any(_condition_holds(inputs, c) for c in any_conds):
            continue
        fired.append(FiredRule(rule["id"], rule["version"], rule["esi"], rule["reason"], rule["source"]))
    return fired


def rule_floor(fired: List[FiredRule]) -> Optional[int]:
    return min((r.esi for r in fired), default=None)


def run_boundary_tests() -> List[str]:
    """Execute every rule's embedded boundary_tests. Returns human-readable failures (empty = all good)."""
    failures: List[str] = []
    for rule in load_ruleset()["rules"]:
        tests = rule.get("boundary_tests", [])
        if not any(t["fires"] for t in tests) or not any(not t["fires"] for t in tests):
            failures.append(f"{rule['id']}: needs at least one firing and one non-firing boundary test")
        for t in tests:
            got = rule["id"] in {r.id for r in evaluate_rules(t["inputs"])}
            if got != t["fires"]:
                failures.append(f"{rule['id']} v{rule['version']}: {t['inputs']} expected fires={t['fires']}, got {got}")
    return failures
