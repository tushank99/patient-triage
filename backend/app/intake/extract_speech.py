"""
Structured extraction from a (verified or draft) intake transcript: symptoms, negated symptoms, duration, severity,
onset, and history mentions (medications, allergies). Deterministic: each field carries the evidence span and a
confidence that combines the pattern's reliability with the speech recogniser's word confidences for that span.
"""
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from backend.app.clinical.facts import drug_data

SYMPTOMS: Dict[str, List[str]] = {
    "chest pain": ["chest pain", "chest pressure", "chest tightness", "pain in my chest", "pain in the chest"],
    "shortness of breath": ["shortness of breath", "short of breath", "trouble breathing", "can't breathe", "cannot breathe", "breathless", "winded"],
    "headache": ["headache", "head pain", "migraine"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "light-headed", "vertigo"],
    "nausea": ["nausea", "nauseous", "feel sick"],
    "vomiting": ["vomiting", "vomited", "throwing up", "threw up"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "tummy pain", "pain in my stomach"],
    "fever": ["fever", "feverish", "high temperature", "chills"],
    "cough": ["cough", "coughing"],
    "fall": ["fell", "fall", "fallen", "tripped"],
    "head injury": ["hit my head", "hit her head", "hit his head", "head injury", "banged my head"],
    "weakness": ["weakness", "weak", "can't move my arm", "can't move my leg"],
    "slurred speech": ["slurred speech", "slurring", "trouble speaking"],
    "palpitations": ["palpitations", "racing heart", "heart racing", "heart pounding"],
    "back pain": ["back pain", "pain in my back"],
    "sore throat": ["sore throat", "throat pain"],
    "rash": ["rash", "hives"],
    "bleeding": ["bleeding", "blood in"],
    "seizure": ["seizure", "fit", "convulsion"],
    "confusion": ["confused", "confusion", "disoriented"],
    "leg swelling": ["leg swelling", "swollen leg", "swollen legs", "ankle swelling"],
    "urinary symptoms": ["burning when i pee", "burning urination", "painful urination", "dysuria"],
}
NEGATION = re.compile(r"\b(no|not|denies|denied|without|never|don't have|doesn't have|haven't had|negative for)\b[^.;,]{0,40}$", re.I)
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "a": 1, "an": 1, "couple of": 2, "few": 3, "several": 3}
UNITS = {"minute": "minutes", "min": "minutes", "hour": "hours", "hr": "hours", "day": "days", "week": "weeks", "month": "months", "year": "years"}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, NUM_WORDS), key=len, reverse=True)) + r")"
DURATION = re.compile(r"\b(?:for|since|over|past|last|x)\s*(?:the\s+)?(?:past\s+|last\s+)?" + _NUM + r"\s*(minute|min|hour|hr|day|week|month|year)s?\b", re.I)
SINCE = re.compile(r"\bsince\s+(this morning|last night|yesterday|today|this afternoon|last week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
AGO = re.compile(r"\b" + _NUM + r"\s*(minute|min|hour|hr|day|week|month|year)s?\s+ago\b", re.I)
PAIN_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
PAIN_SCORE = re.compile(r"\b(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:/|out of|over)\s*(?:10|ten)\b", re.I)
SEVERITY_WORDS = re.compile(r"\b(worst (?:pain|headache) (?:of my life|ever)|excruciating|severe|very bad|terrible|moderate|mild|slight)\b", re.I)
ONSET = re.compile(r"\b(sudden(?:ly)?|all of a sudden|out of nowhere|gradual(?:ly)?|slowly getting worse|came on slowly)\b", re.I)
ALLERGY = re.compile(r"\ballergic to ([a-z][a-z -]{2,30}?)(?:[.,;]| and |$)|\b([a-z]+) allergy\b", re.I)


@dataclass
class Field:
    name: str  # symptom | negated_symptom | duration | pain_score | severity | onset | medication | allergy
    value: str
    evidence: str
    start: int
    end: int
    confidence: float
    low_confidence: bool = False
    normalized: Optional[dict] = None


def _span_conf(start: int, end: int, words: Sequence[dict]) -> Optional[float]:
    """Mean recogniser probability of the words overlapping [start, end) of the transcript text."""
    probs = [w["probability"] for w in words if w.get("char_start") is not None and w["char_start"] < end and w["char_end"] > start]
    return sum(probs) / len(probs) if probs else None


def extract(text: str, words: Sequence[dict] = (), min_conf: float = 0.6) -> List[Field]:
    low = text.lower()
    out: List[Field] = []

    def add(name: str, value: str, start: int, end: int, rule_conf: float, normalized: Optional[dict] = None) -> None:
        asr = _span_conf(start, end, words)
        conf = round(rule_conf * (asr if asr is not None else 1.0), 3)
        out.append(Field(name, value, text[start:end], start, end, conf, conf < min_conf, normalized))

    for canon, phrases in SYMPTOMS.items():
        for p in phrases:
            for m in re.finditer(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low):
                negated = bool(NEGATION.search(low[max(0, m.start() - 45): m.start()]))
                add("negated_symptom" if negated else "symptom", canon, m.start(), m.end(), 0.85 if negated else 0.95)
                break
            else:
                continue
            break
    for m in DURATION.finditer(low):
        n = NUM_WORDS.get(m[1].lower(), None) or float(m[1])
        add("duration", f"{n:g} {UNITS[m[2].lower()]}", m.start(), m.end(), 0.95, {"value": n, "unit": UNITS[m[2].lower()]})
    for m in AGO.finditer(low):
        n = NUM_WORDS.get(m[1].lower(), None) or float(m[1])
        add("duration", f"{n:g} {UNITS[m[2].lower()]}", m.start(), m.end(), 0.9, {"value": n, "unit": UNITS[m[2].lower()]})
    for m in SINCE.finditer(low):
        add("duration", f"since {m[1]}", m.start(), m.end(), 0.85, {"since": m[1]})
    for m in PAIN_SCORE.finditer(low):
        v = int(m[1]) if m[1].isdigit() else PAIN_WORDS[m[1].lower()]
        if 0 <= v <= 10:
            add("pain_score", str(v), m.start(), m.end(), 0.97, {"value": v})
    for m in SEVERITY_WORDS.finditer(low):
        add("severity", m[1].lower(), m.start(), m.end(), 0.8)
    for m in ONSET.finditer(low):
        add("onset", "sudden" if m[1].lower().startswith(("sudden", "all of", "out of")) else "gradual", m.start(), m.end(), 0.85)
    drugs = set(drug_data()["anticoagulants"]) | {d for ds in drug_data()["allergy_classes"].values() for d in ds}
    for d in sorted(drugs):
        for m in re.finditer(r"(?<![a-z])" + re.escape(d) + r"(?![a-z])", low):
            ctx = low[max(0, m.start() - 20): m.start()]
            if "allerg" not in ctx:
                add("medication", d, m.start(), m.end(), 0.9)
            break
    for m in ALLERGY.finditer(low):
        sub = (m[1] or m[2] or "").strip()
        if sub and sub not in ("no", "any", "known"):
            add("allergy", sub, m.start(), m.end(), 0.85)
    return out


def fields_as_dicts(fields: Sequence[Field]) -> List[dict]:
    return [asdict(f) for f in fields]
