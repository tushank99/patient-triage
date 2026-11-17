"""
Patient history retrieval for triage.

Only READY documents count. Retrieval is patient-scoped by construction (every query carries patient_id).
  - safety-critical documents (allergies, medications, problem list) are ALWAYS included, never left to ranking;
  - narrative documents come from the Phase 4 retriever (hybrid dense + BM25, query expansion, optional
    cross-encoder rerank, parent-section context). The snippet is the compressed section around the best hit.
  - LOW-CONFIDENCE FALLBACK: when even the best passage is below the calibrated relevance threshold, NO
    narrative documents are sent (irrelevant history misleads the LLM more than none); pinned documents still are.
  - if the patient has no indexed chunks yet (e.g. stale index), keyword overlap + recency is the fallback.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.clinical.facts import ClinicalFacts, SourceDoc, extract_facts
from backend.app.models import ClinicalDocument, DocumentChunk, Patient

PINNED_DOC_TYPES = ("allergy_record", "medication_list", "problem_list")
MAX_NARRATIVE_DOCS = 3
MAX_CHARS_PER_DOC = 700
CHUNK_CANDIDATES = 12

_STOPWORDS = {
    "the", "and", "for", "with", "without", "from", "this", "that", "patient", "reports", "since",
    "past", "after", "into", "has", "have", "had", "was", "were", "are", "not", "but", "of", "to",
    "in", "on", "at", "a", "an", "or", "is", "x2", "days", "day", "hours", "mild", "left", "right",
}

logger = logging.getLogger(__name__)


@dataclass
class HistoryDoc:
    id: str
    doc_type: str
    title: str
    doc_date: Optional[Any]
    content: str
    score: float = 0.0
    section: Optional[str] = None
    page: Optional[int] = None

    @property
    def snippet(self) -> str:
        return self.content[:MAX_CHARS_PER_DOC]


@dataclass
class PatientHistory:
    documents: List[HistoryDoc] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    retrieval: str = "none"  # hybrid/vector/bm25 config name | keyword | low_confidence | none
    low_confidence: bool = False
    top_relevance: Optional[float] = None
    threshold: Optional[float] = None
    n_on_file: int = 0  # READY documents for the patient, whatever retrieval chose to send
    clinical: ClinicalFacts = field(default_factory=ClinicalFacts)
    degraded: List[str] = field(default_factory=list)  # e.g. "retrieval_degraded": embedder/index failed, keyword fallback used  # extracted from ALL documents, not just retrieved ones

    @property
    def has_history(self) -> bool:
        """'We have records for this patient' (also the ML feature). Not 'something relevant was retrieved'."""
        return self.n_on_file > 0 or bool(self.documents)


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z][a-z-]{2,}", text.lower()) if t not in _STOPWORDS}


def derive_facts(history_facts: Optional[dict]) -> Dict[str, Any]:
    if not history_facts:
        return {}
    meds = history_facts.get("medications", [])
    return {
        "on_anticoagulant": any(m.get("status") == "active" and m.get("class") == "anticoagulant" for m in meds),
        "allergies": [a.get("substance") for a in history_facts.get("allergies", [])],
    }


def count_documents(db: Session, patient: Patient) -> int:
    return db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == patient.id, ClinicalDocument.status == "READY").count()


def retrieve_history(db: Session, patient: Optional[Patient], complaint: str) -> PatientHistory:
    if patient is None:
        return PatientHistory()
    rows = db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == patient.id, ClinicalDocument.status == "READY").all()
    if not rows:
        return PatientHistory()

    pinned: List[HistoryDoc] = []
    narrative: Dict[str, HistoryDoc] = {}
    for r in rows:
        doc = HistoryDoc(r.id, r.doc_type, r.title, r.doc_date, r.content)
        (pinned.append(doc) if r.doc_type in PINNED_DOC_TYPES else narrative.__setitem__(r.id, doc))

    low, top_rel, threshold, method = False, None, None, "keyword"
    degraded: List[str] = []
    if narrative:
        from backend.app.retrieval.retriever import RetrievalConfig, retrieve

        cfg = RetrievalConfig.from_settings()
        try:
            result = retrieve(db, patient.id, complaint, cfg)
        except Exception as e:  # embedder/index outage must not block triage: fall back to keyword overlap
            logger.error("Retrieval failed (%s); using keyword fallback", type(e).__name__)
            from backend.app.ops.observability import DEGRADED

            DEGRADED.labels("retrieval_degraded").inc()
            degraded.append("retrieval_degraded")
            result = None
        if result is not None and result.ranked_chunk_ids:
            method, top_rel, threshold = cfg.name, result.top_relevance, result.threshold
            if result.low_confidence:
                low, method = True, "low_confidence"
                narrative = {}
            else:
                order = {}
                for p in result.passages:
                    if p.document_id in narrative and p.document_id not in order:
                        d = narrative[p.document_id]
                        d.content = p.context_text if not p.section else f"[{p.section}] {p.context_text}"
                        d.section, d.page, d.score = p.section, p.page_start, p.relevance
                        order[p.document_id] = len(order)
                narrative = {k: narrative[k] for k in order}  # documents with no passage in the top-k are not sent
    if method == "keyword":
        query = _tokens(complaint)
        for d in narrative.values():
            d.score = float(len(query & _tokens(d.title + " " + d.content)))

    ranked = sorted(narrative.values(), key=lambda d: (d.score, d.doc_date.isoformat() if d.doc_date else ""), reverse=True)
    pinned.sort(key=lambda d: PINNED_DOC_TYPES.index(d.doc_type))
    if method != "keyword":
        ranked = sorted(narrative.values(), key=lambda d: -d.score)
    documents = pinned + ranked[:MAX_NARRATIVE_DOCS]

    # Deterministic facts from every READY document (structured lists AND notes), with temporal status and conflicts.
    clinical = extract_facts([SourceDoc(r.id, r.doc_type, r.doc_date, r.content) for r in rows])
    # A conflict must be checkable: make sure the documents it cites are among those shown (and citable).
    shown = {d.id for d in documents}
    by_id = {r.id: r for r in rows}
    for c in clinical.conflicts:
        for doc_id in c.sources:
            if doc_id and doc_id not in shown and doc_id in by_id:
                r = by_id[doc_id]
                documents.append(HistoryDoc(r.id, r.doc_type, r.title, r.doc_date, r.content))
                shown.add(doc_id)
    facts = derive_facts(patient.history_facts)
    facts["on_anticoagulant"] = bool(facts.get("on_anticoagulant")) or clinical.on_anticoagulant  # uploaded notes count too
    return PatientHistory(documents=documents, facts=facts, retrieval=method, low_confidence=low, top_relevance=top_rel,
                          threshold=threshold, n_on_file=len(rows), clinical=clinical, degraded=degraded)
