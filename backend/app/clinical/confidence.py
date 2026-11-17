"""
Three separate confidence signals plus an explicit missing-data strategy.

They are deliberately NOT combined: a high model probability on a record with no SpO2 is not "high
confidence", and a rich history does not make a weak model stronger. The UI shows all three.
"""
from typing import Dict, List, Optional, Sequence

from backend.app.schemas import ConfidenceBreakdown

# Weight of each vital in data_completeness (sums to 1.0).
VITAL_WEIGHTS: Dict[str, float] = {
    "oxygen_saturation": 0.25, "heart_rate": 0.20, "systolic_bp": 0.20,
    "respiratory_rate": 0.15, "temperature_celsius": 0.10, "pain_score": 0.10,
}

# Vitals that must be measured for a given complaint category. Missing one caps completeness at 0.5
# and forces clinician review (the old engine only subtracted points).
CRITICAL_VITALS: Dict[str, Sequence[str]] = {
    "CARDIAC": ("systolic_bp", "heart_rate"),
    "RESPIRATORY": ("oxygen_saturation", "respiratory_rate"),
    "NEURO": ("systolic_bp", "heart_rate"),
    "INFECTION": ("temperature_celsius", "heart_rate"),
    "TRAUMA_MAJOR": ("systolic_bp", "heart_rate"),
}

_CATEGORY_KEYWORDS = {
    "CARDIAC": ("chest pain", "palpitation", "crushing", "radiating to left arm", "tearing"),
    "RESPIRATORY": ("short of breath", "shortness of breath", "winded", "wheez", "cough", "breath", "asthma", "copd"),
    "NEURO": ("weakness", "slurred", "seizure", "unresponsive", "confus", "stroke", "headache"),
    "INFECTION": ("fever", "febrile", "stiff neck", "sore throat", "infection"),
    "TRAUMA_MAJOR": ("mvc", "motor vehicle", "crush", "stab", "gunshot", "head injury"),
}

LABELS = {
    "oxygen_saturation": "SpO2", "heart_rate": "heart rate", "systolic_bp": "blood pressure",
    "respiratory_rate": "respiratory rate", "temperature_celsius": "temperature", "pain_score": "pain score",
}


def infer_category(complaint: str) -> str:
    """Keyword fallback used only when the LLM did not return a category."""
    text = complaint.lower()
    for category, words in _CATEGORY_KEYWORDS.items():
        if any(w in text for w in words):
            return category
    return "OTHER"


def data_completeness(vitals: Dict[str, Optional[float]], category: str) -> tuple[float, List[str]]:
    score = sum(w for name, w in VITAL_WEIGHTS.items() if vitals.get(name) is not None)
    missing = [LABELS[v] for v in CRITICAL_VITALS.get(category, ()) if vitals.get(v) is None]
    if missing:
        score = min(score, 0.5)
    return round(score, 3), missing


def retrieval_confidence(
    registered: bool, doc_types: Sequence[str], newest_doc_age_days: Optional[float]
) -> tuple[Optional[float], str]:
    """Heuristic until Phase 2 measures retrieval quality (Recall@K etc.)."""
    if not registered:
        return None, "Walk-in: no record on file to retrieve."
    if not doc_types:
        return 0.0, "Registered patient with no history documents on file."
    pinned = {"allergy_record", "medication_list", "problem_list"}
    score = min(len(doc_types), 4) / 4 * 0.5
    score += 0.3 * len(pinned & set(doc_types)) / len(pinned)
    if newest_doc_age_days is not None and newest_doc_age_days <= 730:
        score += 0.2
    return round(min(score, 1.0), 3), f"{len(doc_types)} document(s) retrieved; newest {'unknown' if newest_doc_age_days is None else f'{newest_doc_age_days / 365:.1f}y old'}."


def build_confidence(
    vitals: Dict[str, Optional[float]],
    category: str,
    model_probability: Optional[float],
    registered: bool,
    doc_types: Sequence[str],
    newest_doc_age_days: Optional[float],
    model_llm_agree: Optional[bool],
    low_confidence_retrieval: bool = False,
) -> ConfidenceBreakdown:
    completeness, missing = data_completeness(vitals, category)
    r_score, r_note = retrieval_confidence(registered, doc_types, newest_doc_age_days)
    if low_confidence_retrieval:
        r_note += " No narrative note matched this complaint confidently, so only allergies, medications and problem list were used."
    return ConfidenceBreakdown(
        model_probability=None if model_probability is None else round(model_probability, 4),
        data_completeness=completeness,
        retrieval_confidence=r_score,
        retrieval_note=r_note,
        missing_critical=missing,
        model_llm_agree=model_llm_agree,
    )
