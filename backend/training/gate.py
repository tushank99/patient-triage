"""
The promotion gate: pure functions over model cards, so every rule is unit-tested.

A check has a group ("parity", "golden", "regression", ...). The channel decides which groups BLOCK;
the rest are reported as warnings. Failing a blocking check means the model does not ship.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gate_config.json")
DEPLOYED = "xgb_rules"  # the decision that actually ships: model ESI combined with the rule floor


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Check:
    group: str
    name: str
    status: str  # pass | fail | skip
    detail: str
    blocking: bool = False


@dataclass
class GateResult:
    channel: str
    checks: List[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c.status == "fail" and c.blocking for c in self.checks)

    def table(self) -> str:
        rows = [f"{'BLOCK' if c.blocking and c.status == 'fail' else c.status.upper():<6} {c.group:<11} {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(rows) + f"\n=> {'PASS' if self.passed else 'FAIL'} (channel: {self.channel})"

    def as_dict(self) -> Dict[str, Any]:
        return {"channel": self.channel, "passed": self.passed, "checks": [c.__dict__ for c in self.checks]}


def _num(x: Optional[float], digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def evaluate_gate(candidate: Dict, shipped: Optional[Dict], channel: str, signoff: Optional[str] = None,
                  golden: Optional[Dict] = None, config: Optional[Dict] = None) -> GateResult:
    cfg = config or load_config()
    ch = cfg["channels"][channel]
    blocking_groups = set(ch["blocking"])
    res = GateResult(channel)

    def add(group: str, name: str, ok: Optional[bool], detail: str, always_block: bool = False) -> None:
        status = "skip" if ok is None else ("pass" if ok else "fail")
        res.checks.append(Check(group, name, status, detail, blocking=always_block or group in blocking_groups))

    # -- channel policy (always blocking)
    src = candidate["label_source"]
    add("channel", "label source allowed", src in ch["allowed_label_sources"],
        f"{src!r} for the {channel!r} channel (allowed: {ch['allowed_label_sources']})", always_block=True)
    if ch["requires_signoff"]:
        add("channel", "clinical sign-off recorded", bool(signoff), f"signed off by {signoff!r}" if signoff else "pass --signoff 'Name, role'", always_block=True)

    test = candidate["test"]
    deployed = test[DEPLOYED]["overall"]
    rules_only = test["rules_only"]["overall"]
    logreg = test["logreg"]["overall"]

    # -- parity (server ONNX == joblib, calibration included)
    p = candidate["parity"]
    pc = cfg["parity"]
    add("parity", "ONNX vs joblib", p["max_abs_prob_diff"] <= pc["max_abs_prob_diff"] and p["esi_mismatches"] <= pc["max_esi_mismatches"],
        f"max |dp|={p['max_abs_prob_diff']:.2e} over {p['n']} vectors, {p['esi_mismatches']} ESI mismatches")

    # -- golden clinical scenarios (dataset-independent)
    if golden is None:
        add("golden", "golden scenarios", None, "not run")
    else:
        floor = golden["baseline"] - cfg["golden"]["tolerance"]
        add("golden", "golden exact-match", golden["exact_match"] >= floor,
            f"{golden['exact_match']:.2f} ({golden['hits']}/{golden['n']}) vs committed baseline {golden['baseline']:.2f}")
        add("golden", "no golden under-triage", golden["under_triaged"] == 0, f"{golden['under_triaged']} critical scenario(s) under-triaged")

    # -- absolute thresholds
    a = cfg["absolute"]
    has = deployed.get("high_acuity_sensitivity")
    add("absolute", "high-acuity sensitivity", None if has is None else has >= a["min_high_acuity_sensitivity"],
        f"{_num(has)} (>= {a['min_high_acuity_sensitivity']}) on {deployed.get('high_acuity_n')} ESI 1-2 patients")
    add("absolute", "severe under-triage", deployed["severe_under_triage"] <= a["max_severe_under_triage"],
        f"{_num(deployed['severe_under_triage'])} (<= {a['max_severe_under_triage']})")

    # -- calibration
    ece = test["xgb_calibrated"]["overall"].get("ece")
    add("calibration", "ECE after calibration", None if ece is None else ece <= cfg["calibration"]["max_ece"], f"{_num(ece)} (<= {cfg['calibration']['max_ece']})")

    # -- subgroups: pediatric / geriatric must not be served much worse than everyone else
    sc = cfg["subgroups"]
    for name in sc["names"]:
        g = test[DEPLOYED]["subgroups"].get(name)
        if not g or g["n"] < sc["min_n"] or g.get("high_acuity_sensitivity") is None or has is None:
            add("subgroups", name, None, "too few patients to judge" if g else "no such subgroup")
            continue
        gap = has - g["high_acuity_sensitivity"]
        add("subgroups", name, gap <= sc["max_high_acuity_sensitivity_gap"],
            f"high-acuity sensitivity {_num(g['high_acuity_sensitivity'])} vs overall {_num(has)} (gap {gap:+.3f}, n={g['n']})")

    # -- baselines
    b = cfg["baselines"]
    if b["xgb_rules_must_match_rules_only_high_acuity_sensitivity"]:
        ro = rules_only.get("high_acuity_sensitivity")
        add("baselines", "not worse than rules-only (high-acuity)", None if None in (has, ro) else has + 1e-9 >= ro,
            f"shipped combo {_num(has)} vs rules-only {_num(ro)}")
    add("baselines", "beats logistic regression (macro-F1)", deployed["macro_f1"] >= logreg["macro_f1"] + b["xgb_rules_macro_f1_must_beat_logreg_by"],
        f"{_num(deployed['macro_f1'])} vs logistic {_num(logreg['macro_f1'])}")

    # -- regression vs the currently shipped model
    r = cfg["regression"]
    if shipped is None:
        add("regression", "vs shipped model", None, "no shipped model card yet")
    elif r["applies_only_when_same_dataset"] and shipped.get("dataset_sha256") != candidate.get("dataset_sha256"):
        add("regression", "vs shipped model", None, "shipped model used a different dataset; metrics are not comparable (golden check still applies)")
    else:
        old = shipped["test"][DEPLOYED]["overall"]
        oldece = shipped["test"]["xgb_calibrated"]["overall"].get("ece")
        if has is not None and old.get("high_acuity_sensitivity") is not None:
            add("regression", "high-acuity sensitivity", has >= old["high_acuity_sensitivity"] - r["max_drop_high_acuity_sensitivity"],
                f"{_num(has)} vs shipped {_num(old['high_acuity_sensitivity'])}")
        add("regression", "severe under-triage", deployed["severe_under_triage"] <= old["severe_under_triage"] + r["max_rise_severe_under_triage"],
            f"{_num(deployed['severe_under_triage'])} vs shipped {_num(old['severe_under_triage'])}")
        add("regression", "macro-F1", deployed["macro_f1"] >= old["macro_f1"] - r["max_drop_macro_f1"], f"{_num(deployed['macro_f1'])} vs shipped {_num(old['macro_f1'])}")
        if ece is not None and oldece is not None:
            add("regression", "ECE", ece <= oldece + r["max_rise_ece"], f"{_num(ece)} vs shipped {_num(oldece)}")
    return res
