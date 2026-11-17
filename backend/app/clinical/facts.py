"""
Deterministic clinical fact extraction, temporal reasoning and conflict detection.

From the patient's documents (structured lists AND free-text notes) we extract:
  conditions   name, documented status (ACTIVE / RESOLVED / HISTORICAL), effective status after temporal rules
  medications  name, dose, class, status, and later "stopped"/"started" events from notes (newest wins)
  allergies    substance, reaction, severity; plus "no known allergies" statements
and detect conflicts:
  history-internal  medication list says ACTIVE but a newer note says STOPPED; allergy record vs a note saying NKDA
  intake vs history intake denies allergies that are documented; intake mentions a drug the patient is allergic to
                    (class cross-reactivity); intake mentions a drug the record says was stopped; intake says
                    "blood thinner" but no active anticoagulant is on file (or the reverse)
Temporal rule: a temporary condition (pregnancy, infection, ...) older than ~9 months is treated as resolved unless
documented ACTIVE recently.

Deterministic on purpose: these facts feed safety rules and are shown to the nurse, so they must be reproducible.
They are ALSO given to the LLM as untrusted data, and LLM claims are checked against the source passages.
"""
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

PATH = os.path.join(os.path.dirname(__file__), "drug_classes.json")


@lru_cache(maxsize=1)
def drug_data() -> Dict:
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Condition:
    name: str
    status: str  # ACTIVE | RESOLVED | HISTORICAL
    effective_status: str  # current | resolved | historical
    since: Optional[str] = None
    resolved: Optional[str] = None
    note: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Medication:
    name: str
    dose: str = ""
    drug_class: str = ""
    status: str = "ACTIVE"  # as documented in the medication list
    effective_status: str = "active"  # after newer notes are applied
    since: Optional[str] = None
    stopped: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Allergy:
    substance: str
    reaction: str = ""
    severity: str = ""
    source: Optional[str] = None


@dataclass
class Conflict:
    kind: str  # medication_status | allergy_documentation | intake_denies_allergy | intake_allergen_exposure | intake_stopped_drug | intake_anticoagulant
    severity: str  # high | moderate
    description: str
    sources: List[str] = field(default_factory=list)


@dataclass
class ClinicalFacts:
    conditions: List[Condition] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    allergies: List[Allergy] = field(default_factory=list)
    nkda_statements: List[str] = field(default_factory=list)  # doc ids that say "no known drug allergies"
    conflicts: List[Conflict] = field(default_factory=list)

    @property
    def active_medications(self) -> List[Medication]:
        return [m for m in self.medications if m.effective_status == "active"]

    @property
    def on_anticoagulant(self) -> bool:
        anticoag = set(drug_data()["anticoagulants"])
        return any(m.drug_class == "anticoagulant" or m.name.lower() in anticoag for m in self.active_medications)

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SourceDoc:
    id: str
    doc_type: str
    doc_date: Optional[date]
    text: str


_PROBLEM = re.compile(r"^- (?P<name>.+?): (?:ACTIVE since (?P<since>[\d-]+)|RESOLVED \((?P<frm>[\d-]+) to (?P<to>[\d?-]+)\)|HISTORICAL event \((?P<when>[\d-]+)\))", re.M)
_MED = re.compile(r"^- (?P<body>.+?) \((?P<cls>[a-z0-9 /-]+)\): (?P<status>ACTIVE|DISCONTINUED)(?: since (?P<since>[\d-]+))?(?: (?P<stopped>\d{4}-\d{2}(?:-\d{2})?))?", re.M)
_ALLERGY = re.compile(r"^- (?P<sub>.+?): (?P<reaction>.+?) \(severity: (?P<sev>\w+)\)", re.M)
_NKDA = re.compile(r"\b(no known (drug )?allergies|nkda|states no known drug allergies|denies (any )?(drug )?allergies)\b", re.I)
_STOPPED = re.compile(r"\b(?P<drug>[A-Za-z][a-z]+(?:-[a-z]+)?)\s+(?:was\s+|has been\s+)?(?:STOPPED|stopped|discontinued|ceased|held)\b")
_STARTED = re.compile(r"\b(?P<drug>[a-z][a-z-]+)\s+\d+(?:\.\d+)?\s*mg\s+(?:was\s+)?started\b", re.I)
_DOC_ALLERGY = re.compile(r"\b(?P<sub>[A-Z][a-z]+) allergy documented\b|\ballergic to (?P<sub2>[a-z]+)\b", re.I)


def _split_med(body: str) -> tuple[str, str]:
    m = re.search(r"\s\d", body)
    return (body[: m.start()].strip(), body[m.start():].strip()) if m else (body.strip(), "")


def _days_between(a: Optional[str], b: date) -> Optional[int]:
    if not a:
        return None
    try:
        parts = [int(x) for x in a.split("-")] + [1, 1]
        return (b - date(parts[0], parts[1], parts[2] if len(a.split("-")) > 2 else 1)).days
    except ValueError:
        return None


def allergy_class(substance: str) -> Optional[str]:
    s = substance.lower()
    for cls, members in drug_data()["allergy_classes"].items():
        if cls in s or any(m in s for m in members):
            return cls
    return None


def extract_facts(docs: Sequence[SourceDoc], today: Optional[date] = None) -> ClinicalFacts:
    today = today or date.today()
    data = drug_data()
    facts = ClinicalFacts()
    meds: Dict[str, Medication] = {}
    list_date: Dict[str, Optional[date]] = {}

    for d in docs:
        for m in _PROBLEM.finditer(d.text):
            status = "ACTIVE" if m["since"] else ("RESOLVED" if m["frm"] else "HISTORICAL")
            eff = {"ACTIVE": "current", "RESOLVED": "resolved", "HISTORICAL": "historical"}[status]
            note = None
            name_l = m["name"].lower()
            age = _days_between(m["since"] or m["frm"] or m["when"], today)
            if status == "ACTIVE" and any(t in name_l for t in data["temporary_conditions"]) and age is not None and age > data["temporary_window_days"]:
                eff, note = "resolved", "temporary condition documented over 9 months ago; treated as resolved"
            facts.conditions.append(Condition(m["name"], status, eff, m["since"] or m["frm"] or m["when"], m["to"], note, d.id))
        for m in _MED.finditer(d.text):
            name, dose = _split_med(m["body"])
            med = Medication(name, dose, m["cls"].replace(" ", "_"), m["status"], "active" if m["status"] == "ACTIVE" else "discontinued",
                             m["since"], m["stopped"], d.id)
            meds[name.lower()] = med
            list_date[name.lower()] = d.doc_date
        for m in _ALLERGY.finditer(d.text):
            facts.allergies.append(Allergy(m["sub"], m["reaction"], m["sev"], d.id))
        if d.doc_type != "allergy_record" and _NKDA.search(d.text):
            facts.nkda_statements.append(d.id)
        elif d.doc_type == "allergy_record" and _NKDA.search(d.text) and not _ALLERGY.search(d.text):
            facts.nkda_statements.append(d.id)

    # Free-text events from notes, applied in date order: newer documentation wins.
    for d in sorted((x for x in docs if x.doc_type not in ("medication_list", "allergy_record", "problem_list")), key=lambda x: x.doc_date or date.min):
        for m in _STOPPED.finditer(d.text):
            key = m["drug"].lower()
            med = meds.get(key)
            if med and med.effective_status == "active":
                listed = list_date.get(key)
                med.effective_status, med.stopped = "discontinued", (d.doc_date.isoformat() if d.doc_date else med.stopped)
                if med.status == "ACTIVE":
                    facts.conflicts.append(Conflict(
                        "medication_status", "moderate",
                        f"{med.name} is ACTIVE on the medication list{f' ({listed})' if listed else ''} but a newer note"
                        f"{f' ({d.doc_date})' if d.doc_date else ''} says it was stopped. Using the newer note.",
                        [med.source or "", d.id]))
        for m in _STARTED.finditer(d.text):
            key = m["drug"].lower()
            if key not in meds:
                meds[key] = Medication(m["drug"].capitalize(), "", "", "NOTE", "active", d.doc_date.isoformat() if d.doc_date else None, None, d.id)
        for m in _DOC_ALLERGY.finditer(d.text):
            sub = (m["sub"] or m["sub2"] or "").capitalize()
            if sub and not any(sub.lower() in a.substance.lower() for a in facts.allergies):
                facts.allergies.append(Allergy(sub, "documented in note", "", d.id))
    facts.medications = list(meds.values())

    if facts.allergies and facts.nkda_statements:
        facts.conflicts.append(Conflict(
            "allergy_documentation", "high",
            f"Allergy record lists {', '.join(a.substance for a in facts.allergies)}, but another document says 'no known drug allergies'. "
            "Trust the allergy record until verified with the patient.",
            sorted({a.source or "" for a in facts.allergies} | set(facts.nkda_statements))))
    return facts


_INTAKE_DENIES = re.compile(r"\b(no (known )?(drug )?allergies|nkda|denies (any )?(drug )?allergies|not allergic to anything)\b", re.I)


def intake_conflicts(facts: ClinicalFacts, complaint: str) -> List[Conflict]:
    data = drug_data()
    text = complaint.lower()
    out: List[Conflict] = []
    if facts.allergies and _INTAKE_DENIES.search(complaint):
        out.append(Conflict("intake_denies_allergy", "high",
                            f"Intake says no allergies, but the record documents: {', '.join(f'{a.substance} ({a.reaction})' for a in facts.allergies)}.",
                            sorted({a.source or "" for a in facts.allergies})))
    for a in facts.allergies:
        cls = allergy_class(a.substance)
        if not cls:
            continue
        hits = []
        for member in data["allergy_classes"][cls]:
            for mm in re.finditer(r"(?<![a-z])" + re.escape(member) + r"(?![a-z])", text):
                before, after = text[max(0, mm.start() - 20): mm.start()], text[mm.end(): mm.end() + 12]
                agrees = re.search(r"allergic to\s+(an?\s+|the\s+)?$|allergy to\s+$|reaction to\s+$", before) or re.match(r"\s*(allergy|allergic|reaction)", after)
                if not agrees:  # "allergic to penicillin" agrees with the record; "took amoxicillin" is an exposure
                    hits.append(member)
        if hits:
            out.append(Conflict("intake_allergen_exposure", "high",
                                f"Intake mentions {hits[0]}, which belongs to a documented allergy class ({a.substance}: {a.reaction}).", [a.source or ""]))
    for m in facts.medications:
        if m.effective_status == "discontinued" and re.search(r"(?<![a-z])" + re.escape(m.name.lower()) + r"(?![a-z])", text):
            out.append(Conflict("intake_stopped_drug", "moderate",
                                f"Intake mentions {m.name}, which the record says was stopped{f' ({m.stopped})' if m.stopped else ''}.", [m.source or ""]))
    says_anticoag = any(p in text for p in data["anticoagulant_phrases"]) or any(
        re.search(r"(?<![a-z])" + re.escape(d) + r"(?![a-z])", text) for d in data["anticoagulants"])
    if says_anticoag and not facts.on_anticoagulant and facts.medications:
        out.append(Conflict("intake_anticoagulant", "moderate",
                            "Intake mentions an anticoagulant but none is active on the medication record. Verify current medications.",
                            sorted({m.source or "" for m in facts.medications})))
    return out


def facts_block(facts: ClinicalFacts, conflicts: Sequence[Conflict], ref_of: Dict[str, str]) -> str:
    """Compact text for the LLM. Source ids are replaced by the D-refs the LLM can cite (or omitted)."""
    def src(doc_id: Optional[str]) -> str:
        return f" [{ref_of[doc_id]}]" if doc_id and doc_id in ref_of else ""

    lines = []
    act = [f"{m.name} {m.dose}".strip() + src(m.source) for m in facts.medications if m.effective_status == "active"]
    stop = [f"{m.name}" + src(m.source) for m in facts.medications if m.effective_status == "discontinued"]
    lines.append("Active medications: " + ("; ".join(act) if act else "none documented"))
    if stop:
        lines.append("Discontinued medications (NOT current): " + "; ".join(stop))
    alg = [f"{a.substance}: {a.reaction}" + src(a.source) for a in facts.allergies]
    lines.append("Allergies: " + ("; ".join(alg) if alg else ("none documented (NKDA)" if facts.nkda_statements else "not documented")))
    cur = [c.name + src(c.source) for c in facts.conditions if c.effective_status == "current"]
    past = [f"{c.name} ({c.effective_status})" + src(c.source) for c in facts.conditions if c.effective_status != "current"]
    lines.append("Current conditions: " + ("; ".join(cur) if cur else "none documented"))
    if past:
        lines.append("Resolved/historical (NOT current): " + "; ".join(past))
    for c in conflicts:
        lines.append(f"CONFLICT ({c.severity}): {c.description}")
    return "\n".join(lines)
