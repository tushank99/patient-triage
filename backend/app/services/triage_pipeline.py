"""
THE assessment pipeline. Every screen goes through run_assessment().

  1. De-identify: names, MRN, contact details and calendar dates never reach the LLM (dates become
     "2.1 years ago"); history documents are labelled D1, D2... instead of real ids; age 90+ is "90+".
  2. ML:     XGBoost on the shared feature schema.
  3. LLM:    structured, validated output (versioned prompt, retry, fallback). Optional.
  4. Rules:  deterministic safety floor.
  5. Fusion: clinical/fusion.py (rules floor > model escalation > LLM/ML base). A human may override later.
  6. Confidence: three separate signals; review flags (clinical/confidence.py).
"""
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

from backend.app.clinical.confidence import build_confidence, infer_category
from backend.app.clinical.facts import facts_block, intake_conflicts
from backend.app.clinical.fusion import fuse
from backend.app.clinical.rules_engine import evaluate_rules, rule_floor, ruleset_version
from backend.app.llm.assessor import LLMCallMeta, LLMInput, LLMOutcome, LLMUnavailable, assess_with_llm
from backend.app.llm.grounding import GroundingReport, validate_claims
from backend.app.llm.security import neutralise
from backend.app.ml.feature_schema import FEATURE_SCHEMA_VERSION
from backend.app.ml.model import MLResult, get_model
from backend.app.schemas import (
    AssessmentComponents,
    Badge,
    ConfidenceBreakdown,
    DecisionStep,
    HistoryContext,
    HistoryDocumentRef,
    FeatureContribution,
    RuleHit,
    Versions,
)
from backend.app.services.history_store import PatientHistory
from backend.app.services.pii import generalize_dates, relative_age_label, scrub_pii

from backend.app.ops.observability import ASSESSMENTS, DEGRADED, stage  # noqa: E402

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "phase1.0"
LOW_MODEL_PROBABILITY = 0.5

LLMFn = Callable[[LLMInput], LLMOutcome]


@dataclass
class AssessmentInput:
    age_years: float
    gender: str
    complaint: str
    vitals: Dict[str, Optional[float]]
    patient_name: Optional[str] = None
    registered: bool = False  # patient already had a record (MRN) before this visit


@dataclass
class Assessment:
    recommended_esi: int
    rationale: str
    risk_factors: List[str]
    badges: List[Badge]
    components: AssessmentComponents
    decision_trace: List[DecisionStep]
    confidence: ConfidenceBreakdown
    requires_review: bool
    review_reasons: List[str]
    history: HistoryContext
    degraded: List[str]
    pii_redactions: int
    versions: Versions
    llm_meta: Optional[LLMCallMeta] = None
    llm_request_text: Optional[str] = None
    claims: list = field(default_factory=list)
    grounding: Optional[dict] = None
    facts: Optional[dict] = None
    security_flags: List[str] = field(default_factory=list)


def _history_block(history: PatientHistory, names: List[str], today: date) -> tuple[str, Dict[str, str], int, Dict[str, str], List[str]]:
    """Returns (text for the LLM, {"D1": real_doc_id}, redaction count, {"D1": exact text shown}, security flags).
    Every passage is de-identified AND scanned for instruction-like text (prompt-injection defence) before it is shown."""
    if not history.has_history:
        return "NO PRIOR HOSPITAL HISTORY FOUND for this patient.", {}, 0, {}, []
    if not history.documents:
        return "Records exist for this patient, but none matched this complaint (low retrieval confidence). Do not assume absent history.", {}, 0, {}, []
    lines, refs, redactions, shown, flags = [], {}, 0, {}, []
    for i, d in enumerate(history.documents, start=1):
        ref = f"D{i}"
        refs[ref] = d.id
        body = scrub_pii(generalize_dates(d.snippet, today), names)
        redactions += body.redactions
        scan = neutralise(body.text)
        if scan.flagged:
            flags.append(f"prompt_injection_suspected:{ref}:{d.id}")
        shown[ref] = scan.text
        lines.append(f"[{ref}] ({d.doc_type}, {relative_age_label(d.doc_date, today)}) {scan.text}")
    return "\n".join(lines), refs, redactions, shown, flags


def generate_badges(age_years: float, v: Dict[str, Optional[float]]) -> List[Badge]:
    """Deterministic visual flags (age-calibrated). Never produced by the LLM."""
    badges: List[Badge] = []
    sbp, spo2, hr = v.get("systolic_bp"), v.get("oxygen_saturation"), v.get("heart_rate")
    if sbp is not None:
        if sbp < 90:
            badges.append(Badge(label=f"Hypotension (BP {sbp:g})", type="danger"))
        elif sbp > 160:
            badges.append(Badge(label=f"Hypertension (BP {sbp:g})", type="warning"))
    if spo2 is not None:
        if spo2 <= 92:
            badges.append(Badge(label=f"Hypoxia (SpO2 {spo2:g}%)", type="danger"))
        elif spo2 >= 95:
            badges.append(Badge(label=f"Normal O2 ({spo2:g}%)", type="success"))
    if hr is not None:
        if age_years < 18:
            if hr > 180:
                badges.append(Badge(label=f"Pediatric Tachycardia ({hr:g})", type="danger"))
        elif hr > 120:
            badges.append(Badge(label=f"Tachycardia ({hr:g})", type="danger"))
        elif 60 <= hr <= 100:
            badges.append(Badge(label=f"Normal HR ({hr:g})", type="success"))
    return badges


def run_assessment(
    inp: AssessmentInput, history: PatientHistory, llm_fn: LLMFn = assess_with_llm, today: Optional[date] = None
) -> Assessment:
    today = today or date.today()
    names = [inp.patient_name] if inp.patient_name else []
    degraded: List[str] = list(history.degraded)

    # 1. de-identify everything that may leave the server
    scrubbed = scrub_pii(generalize_dates(inp.complaint, today), names)
    complaint_scan = neutralise(scrubbed.text)
    history_text, doc_refs, history_redactions, shown, security_flags = _history_block(history, names, today)
    if complaint_scan.flagged:
        security_flags.insert(0, "prompt_injection_suspected:INTAKE")

    # Deterministic facts + conflicts (history-internal from the extractor, intake-vs-history here).
    conflicts = list(history.clinical.conflicts) + intake_conflicts(history.clinical, inp.complaint)
    ref_of = {doc_id: ref for ref, doc_id in doc_refs.items()}
    facts_text = scrub_pii(generalize_dates(facts_block(history.clinical, conflicts, ref_of), today), names).text if history.has_history else ""

    # 2. ML
    features = {"age_years": inp.age_years, **inp.vitals, "has_prior_history": history.has_history}
    model = get_model()
    ml: Optional[MLResult] = None
    if model is None:
        degraded.append("ml_unavailable")
    else:
        try:
            with stage("ml"):
                ml = model.predict(features)
        except Exception as e:
            logger.error("ML inference failed: %s", e)
            degraded.append("ml_unavailable")

    # 3. LLM
    outcome: Optional[LLMOutcome] = None
    try:
        with stage("llm"):
            outcome = llm_fn(LLMInput(
                age_label="90+" if inp.age_years >= 90 else f"{inp.age_years:g}",
                gender=inp.gender, vitals=inp.vitals, complaint=complaint_scan.text, history_block=history_text, facts_block=facts_text,
            ))
    except LLMUnavailable as e:
        degraded.append(e.reason)
    llm = outcome.assessment if outcome else None

    # 4. rules (raw complaint is fine here: it never leaves the server)
    with stage("rules"):
        fired = evaluate_rules({**features, "complaint": inp.complaint, "history": history.facts})
    floor = rule_floor(fired)

    # 5. fusion
    decision = fuse(floor, ml.esi if ml else None, ml.top_probability if ml else None, llm.esi if llm else None)

    # 6. confidence + review flags
    category = llm.category if llm else infer_category(inp.complaint)
    dated = [d.doc_date for d in history.documents if d.doc_date]
    newest_age = (today - max(dated)).days if dated else None
    confidence = build_confidence(
        inp.vitals, category, ml.top_probability if ml else None, inp.registered,
        [d.doc_type for d in history.documents], newest_age,
        None if (llm is None or ml is None) else llm.esi == ml.esi,
        history.low_confidence,
    )

    review: List[str] = []
    if decision.llm_downgrade_blocked:
        review.append(f"Safety rule overrode the AI suggestion (AI ESI {llm.esi}, rules require ESI {floor} or more acute).")
    if decision.disagreement:
        review.append(f"AI (ESI {llm.esi}) and ML model (ESI {ml.esi}) disagree by 2+ levels.")
    if confidence.missing_critical:
        review.append(f"Missing vitals critical for this complaint: {', '.join(confidence.missing_critical)}.")
    if llm is None:
        review.append("AI narrative unavailable; ESI is from the model and safety rules only.")

    # Grounding: every LLM claim must cite what it was shown, and the cited text must support it.
    intake_text = complaint_scan.text + " | vitals: " + ", ".join(f"{k} {v}" for k, v in inp.vitals.items() if v is not None)
    sources = {**shown, "INTAKE": intake_text, "FACTS": facts_text}
    with stage("grounding"):
        report = validate_claims(llm.claims, sources, doc_refs) if llm else GroundingReport([])
    for c in report.problems[:3]:
        review.append(f"AI statement not supported by the record ({c.verdict}): \"{c.text[:90]}\" ({'; '.join(c.reasons)}).")
    for c in conflicts:
        if c.severity == "high":
            review.append(f"Conflict: {c.description}")
    if security_flags:
        review.append("Instruction-like text was found in " + ", ".join(sorted({f.split(':')[1] for f in security_flags}))
                      + " and withheld from the AI (possible prompt injection).")
    if ml is None:
        review.append("ML model unavailable.")
    elif ml.top_probability < LOW_MODEL_PROBABILITY:
        review.append(f"Low model probability ({ml.top_probability:.0%}).")

    if llm is not None:
        rationale = llm.rationale
    else:
        rationale = f"AI narrative unavailable; ESI from {decision.source} and deterministic safety rules only."
    if decision.escalated_by_rules or decision.llm_downgrade_blocked:
        reasons = "; ".join(r.reason for r in fired if r.esi == floor)
        rationale = f"SAFETY RULE ESCALATION to ESI {decision.final_esi} ({reasons}). " + rationale

    risk_factors = list(dict.fromkeys(
        [r.reason for r in fired] + (llm.risk_factors if llm else []) + [f"Not measured: {m}" for m in confidence.missing_critical]
    ))[:6]

    cited = [doc_refs[c] for c in (llm.cited_document_ids if llm else []) if c in doc_refs]  # drop invented refs

    meta = outcome.meta if outcome else None
    for d in degraded:
        DEGRADED.labels(d).inc()
    ASSESSMENTS.labels(decision.source, str(bool(degraded)).lower()).inc()
    return Assessment(
        recommended_esi=decision.final_esi,
        rationale=rationale,
        risk_factors=risk_factors,
        badges=generate_badges(inp.age_years, inp.vitals),
        components=AssessmentComponents(
            ml_esi=ml.esi if ml else None,
            ml_probability=round(ml.top_probability, 4) if ml else None,
            ml_drivers=[FeatureContribution(feature=c.feature, label=c.label, value=c.value, contribution=c.contribution) for c in ml.contributions] if ml else [],
            llm_esi=llm.esi if llm else None,
            llm_category=llm.category if llm else None,
            rule_floor_esi=floor,
            rules_fired=[RuleHit(id=r.id, version=r.version, esi=r.esi, reason=r.reason, source=r.source) for r in fired],
            escalated_by_rules=decision.escalated_by_rules,
            llm_downgrade_blocked=decision.llm_downgrade_blocked,
        ),
        decision_trace=decision.trace,
        confidence=confidence,
        requires_review=bool(review),
        review_reasons=review,
        history=HistoryContext(
            has_history=history.has_history,
            documents=[
                HistoryDocumentRef(id=d.id, doc_type=d.doc_type, title=d.title, doc_date=d.doc_date, snippet=d.snippet,
                                   section=d.section, page=d.page)
                for d in history.documents
            ],
            cited_document_ids=cited,
            retrieval_method=history.retrieval,
            low_confidence=history.low_confidence,
            top_relevance=None if history.top_relevance is None else round(history.top_relevance, 4),
            relevance_threshold=history.threshold,
        ),
        degraded=degraded,
        pii_redactions=scrubbed.redactions + history_redactions,
        versions=Versions(
            pipeline=PIPELINE_VERSION,
            model=ml.model_version if ml else None,
            model_label_source=ml.label_source if ml else None,
            feature_schema=FEATURE_SCHEMA_VERSION,
            rules=ruleset_version(),
            prompt=meta.prompt_version if meta else None,
            llm_provider=meta.provider if meta else None,
            llm_model=meta.model if meta else None,
        ),
        llm_meta=meta,
        llm_request_text=meta.request_text if meta else None,
        claims=[{"text": c.text, "kind": c.kind, "status": c.status, "citations": c.citations, "document_ids": c.document_ids,
                 "verdict": c.verdict, "coverage": c.coverage, "reasons": c.reasons} for c in report.checks],
        grounding={"n_claims": len(report.checks), **report.counts, "score": report.score},
        facts={**history.clinical.as_dict(), "conflicts": [c.__dict__ for c in conflicts]},
        security_flags=security_flags,
    )
