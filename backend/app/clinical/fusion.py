"""
The explicit safety order. One pure function, no I/O, fully unit-tested.

  1. Base suggestion      LLM if available, else ML, else a safe default (ESI 3).
  2. Model escalation     If ML is MORE acute than the base and confident, take the ML level.
                          Models may only raise acuity here; they never lower it.
  3. Rule floor           Deterministic rules set a hard floor: final = min(base, floor).
                          Nothing above (LLM or ML) can place a rules-flagged patient below it.
  4. Human override       Applied later, outside the pipeline (POST /api/override). A clinician may
                          go above or below the AI, but going below the rule floor needs a senior
                          role and is audited as `below_rule_floor`.

ESI numbers: lower = more acute, so "escalate" means a smaller number.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from backend.app.schemas import DecisionStep

SAFE_DEFAULT_ESI = 3
ML_ESCALATION_MIN_PROBABILITY = 0.70
DISAGREEMENT_REVIEW_GAP = 2  # LLM and ML this many levels apart -> clinician review


@dataclass
class FusionDecision:
    final_esi: int
    source: str
    trace: List[DecisionStep] = field(default_factory=list)
    escalated_by_rules: bool = False
    llm_downgrade_blocked: bool = False
    model_escalated: bool = False
    disagreement: bool = False


def fuse(
    rule_floor: Optional[int],
    ml_esi: Optional[int],
    ml_probability: Optional[float],
    llm_esi: Optional[int],
) -> FusionDecision:
    trace: List[DecisionStep] = []

    # 1. base
    if llm_esi is not None:
        base, source = llm_esi, "LLM"
    elif ml_esi is not None:
        base, source = ml_esi, "ML model"
    else:
        base, source = SAFE_DEFAULT_ESI, "safe default"
    trace.append(DecisionStep(stage="base", esi_before=None, esi_after=base, note=f"Base suggestion from {source}."))

    # 2. model escalation (raise only)
    model_escalated = False
    if llm_esi is not None and ml_esi is not None and ml_esi < base and (ml_probability or 0.0) >= ML_ESCALATION_MIN_PROBABILITY:
        trace.append(DecisionStep(
            stage="model_escalation", esi_before=base, esi_after=ml_esi,
            note=f"ML model is more acute (ESI {ml_esi}, p={ml_probability:.2f}); taking the more acute level.",
        ))
        base, model_escalated = ml_esi, True

    # 3. rule floor
    final, escalated = base, False
    # True whenever the LLM wanted a less acute level than the rules allow, whoever raised it in the end.
    blocked = rule_floor is not None and llm_esi is not None and llm_esi > rule_floor
    if rule_floor is not None:
        if rule_floor < base:
            final, escalated = rule_floor, True
            trace.append(DecisionStep(
                stage="rule_floor", esi_before=base, esi_after=rule_floor,
                note="Safety rules require a more acute level than the model output."
                + (" The LLM suggestion was overridden." if blocked else ""),
            ))
        else:
            trace.append(DecisionStep(stage="rule_floor", esi_before=base, esi_after=base, note=f"Rule floor ESI {rule_floor} already satisfied."))

    disagreement = llm_esi is not None and ml_esi is not None and abs(llm_esi - ml_esi) >= DISAGREEMENT_REVIEW_GAP
    return FusionDecision(final, source, trace, escalated, blocked, model_escalated, disagreement)
