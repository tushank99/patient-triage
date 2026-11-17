"""Golden regression set through the real /api/triage endpoint (LLM disabled)."""
import json

from backend.tests.conftest import GOLDEN_PATH, auth, triage_payload


def pytest_generate_tests(metafunc):
    if "scenario" in metafunc.fixturenames:
        scenarios = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["scenarios"]
        metafunc.parametrize("scenario", scenarios, ids=[s["id"] for s in scenarios])


def _master_profiles():
    from backend.scripts.seed_data import load_master

    return {p["mrn"]: p["profile"] for p in load_master()["patients"]}


def test_golden_scenario(client, scenario):
    r = client.post("/api/triage", json=triage_payload(scenario), headers=auth(client))
    assert r.status_code == 200, r.text
    body = r.json()
    fired = {h["id"] for h in body["components"]["rules_fired"]}

    assert not set(scenario["must_fire"]) - fired, f"required safety rules did not fire: {set(scenario['must_fire']) - fired}"
    assert not set(scenario["must_not_fire"]) & fired, f"rules fired that must not: {set(scenario['must_not_fire']) & fired}"
    if scenario["deterministic_max_esi"] is not None:
        assert body["recommended_esi"] <= scenario["deterministic_max_esi"], (
            f"under-triage: got ESI {body['recommended_esi']}, must be <= {scenario['deterministic_max_esi']}"
        )
    expects_history = scenario["mrn"] is not None and _master_profiles()[scenario["mrn"]] != "none"
    assert body["history"]["has_history"] is expects_history
    assert "llm_not_configured" in body["degraded"]
    assert body["requires_clinician_review"] is True  # no LLM -> always flagged for review


def test_deterministic_exact_match_does_not_regress(client, golden):
    hits = sum(
        client.post("/api/triage", json=triage_payload(s), headers=auth(client)).json()["recommended_esi"] == s["expected_esi"]
        for s in golden["scenarios"]
    )
    rate = hits / len(golden["scenarios"])
    print(f"\nDeterministic exact-match vs expected_esi: {hits}/{len(golden['scenarios'])} = {rate:.2f}")
    assert rate >= golden["regression_baseline"]["deterministic_exact_match_min"]
