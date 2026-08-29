from backend.app.services.pii_service import scrub_pii
from backend.app.services.rules_engine import evaluate_guardrails
from backend.app.services.ml_scorer import calculate_mock_ml_score
from backend.app.services.llm_agent import extract_and_refine

async def generate_triage_recommendation(age: float, vitals: dict, complaint: str, has_history: bool) -> dict:
    # 1. Scrub PII
    safe_complaint = scrub_pii(complaint)
    
    # 2. Run Guardrails (Deterministic)
    flags = evaluate_guardrails(age, vitals, safe_complaint)
    
    # 3. Run Tabular ML (Probabilistic)
    ml_result = calculate_mock_ml_score(vitals)
    
    # 4. Run Multi-Agent LLM
    llm_result = await extract_and_refine(safe_complaint, flags)
    
    # --- FUSION LOGIC ---
    final_esi = ml_result["ml_esi"]
    rationale = llm_result.get("rationale", "")
    is_escalated = False
    
    # Conflict Resolution: LLM overrides ML if critical text detected
    if llm_result.get("critical_semantics_detected") and final_esi > 2:
        final_esi = 2
        is_escalated = True
        rationale += " | ESCALATED: NLP detected high-risk text."
        
    # Guardrails override everything
    if flags:
        highest_rule_esi = min([f["level"] for f in flags]) # Lower ESI = Higher Acuity
        if highest_rule_esi < final_esi:
            final_esi = highest_rule_esi
            is_escalated = True
            rule_reasons = ", ".join([f["reason"] for f in flags])
            rationale = f"MANDATORY ESCALATION ({rule_reasons}). " + rationale

    # --- CONFIDENCE CALCULATION ---
    # Missing history or missing vitals drops confidence
    confidence = 0.95
    if not has_history:
        confidence -= 0.15
    if not vitals.get("heart_rate") or not vitals.get("systolic_bp"):
        confidence -= 0.20
        
    # Prevent Nurse Override Fatigue
    if confidence < 0.70:
        rationale = " LOW CONFIDENCE (Missing Data): Manual Assessment Highly Recommended. " + rationale

    return {
        "suggested_esi": final_esi,
        "confidence": round(confidence, 2),
        "rationale": rationale,
        "key_drivers": ml_result["shap_drivers"],
        "is_escalated_failsafe": is_escalated
    }
