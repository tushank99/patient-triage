from datetime import date

from backend.app.clinical.rules_engine import evaluate_rules, load_ruleset, rule_floor, run_boundary_tests
from backend.app.services.pii import generalize_dates, scrub_pii


def fired(**inputs):
    return {r.id for r in evaluate_rules(inputs)}


# ---------------------------------------------------------------- rules


def test_every_rule_has_provenance_and_passing_boundary_tests():
    rules = load_ruleset()["rules"]
    for r in rules:
        assert r["source"] and r["version"] >= 1 and r["status"] and r["boundary_tests"], r["id"]
    assert len({r["id"] for r in rules}) == len(rules)
    assert run_boundary_tests() == []


def test_boundary_test_runner_catches_a_wrong_threshold(monkeypatch):
    from backend.app.clinical import rules_engine

    broken = {"ruleset_version": "x", "rules": [{
        "id": "R-X", "version": 1, "esi": 2, "reason": "r", "source": "s", "status": "t",
        "all": [{"field": "oxygen_saturation", "op": "<", "value": 92}],  # was <= 92
        "boundary_tests": [{"inputs": {"oxygen_saturation": 92}, "fires": True}, {"inputs": {"oxygen_saturation": 95}, "fires": False}],
    }]}
    monkeypatch.setattr(rules_engine, "load_ruleset", lambda: broken)
    assert any("R-X" in f for f in rules_engine.run_boundary_tests())


def test_hypoxia_boundary():
    assert "R-HYPOXIA" in fired(age_years=40, oxygen_saturation=92)
    assert "R-HYPOXIA" not in fired(age_years=40, oxygen_saturation=93)


def test_missing_values_never_fire():
    assert fired(age_years=40) == set()
    assert rule_floor(evaluate_rules({"age_years": 40})) is None


def test_infant_vitals_do_not_trip_adult_thresholds():
    ids = fired(age_years=0.25, heart_rate=165, respiratory_rate=48, systolic_bp=80)
    assert not ids & {"R-HR-EXTREME-ADULT", "R-HR-EXTREME-INFANT", "R-RR-ABNORMAL-ADULT", "R-HYPOTENSION-ADULT"}


def test_headache_is_not_head_trauma():
    assert "R-GERI-FALL" not in fired(age_years=80, complaint="worsening headache for two days")
    assert "R-GERI-FALL" in fired(age_years=80, complaint="fell in the bathroom")


def test_anticoagulant_rule_needs_history():
    assert "R-ANTICOAG-HEAD-INJURY" not in fired(age_years=50, complaint="fell and hit head")
    assert "R-ANTICOAG-HEAD-INJURY" in fired(age_years=50, complaint="fell and hit head", history={"on_anticoagulant": True})
    assert "R-ANTICOAG-HEAD-INJURY" not in fired(age_years=50, complaint="fell and hit head", history={"on_anticoagulant": False})


def test_floor_is_most_acute_rule():
    rules = evaluate_rules({"age_years": 40, "heart_rate": 170, "oxygen_saturation": 88})
    assert rule_floor(rules) == 1


# ---------------------------------------------------------------- PII


def test_name_parts_use_word_boundaries():
    out = scrub_pii("John Doe does not have a doe-eyed look", ["Doe, John"]).text
    assert "John" not in out and " does " in out


def test_structured_identifiers():
    out = scrub_pii("SSN 123-45-6789, phone (555) 123-4567, mail a.b@x.org, MRN-A3F9, born 03/04/1980").text
    for leaked in ("123-45-6789", "555", "a.b@x.org", "A3F9", "1980"):
        assert leaked not in out


def test_clinical_numbers_survive():
    out = scrub_pii("BP 120/80, HR 110, SpO2 89%, pain 8/10, symptoms since 3/4").text
    assert "120/80" in out and "8/10" in out and "3/4" in out


def test_honorific_names_but_not_eponyms():
    out = scrub_pii("Seen by Dr. Alvarez for Parkinson's; Dr. Parkinson's protocol applies").text
    assert "Alvarez" not in out and "Parkinson" in out


def test_dates_become_relative_phrases():
    out = generalize_dates("Pregnancy 2021-02 resolved. Discharged 2026-08-20. Born 1999-13-45", date(2026, 11, 15))
    assert "2021" not in out and "2026-08" not in out
    assert "years ago" in out and "months ago" in out
    assert "1999-13-45" in out  # invalid dates are left for the scrubber, not mangled
