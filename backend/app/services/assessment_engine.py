import os
import json
import logging
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from backend.app.core.config import settings

logger = logging.getLogger(__name__)
llm_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1" if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("gsk_") else None
)

# 1. Load the trained XGBoost model into memory on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_triage_model.joblib")
triage_model = None
if os.path.exists(MODEL_PATH):
    try:
        triage_model = joblib.load(MODEL_PATH)
        logger.info("XGBoost triage model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
else:
    logger.warning(f"XGBoost model not found at {MODEL_PATH}. Will use fallback.")

def run_real_xgboost_inference(patient: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    Performs multi-class inference using the trained XGBoost classifier.
    Returns: (predicted_esi, top_class_probability, margin)
    """
    if triage_model is None:
        return 4, 0.75, 0.1 # Fallback dummy values

    # Helper to safely extract floats, providing clinical defaults if missing
    def _get(key, default):
        val = patient.get(key)
        return float(val) if val is not None else default

    # Must match the exact feature order from the training script
    features = np.array([[
        _get("age", 35.0),
        _get("hr", 75.0),
        _get("bpSys", 120.0),
        _get("bpDia", 80.0),
        _get("spo2", 98.0),
        _get("temp", 37.0),
        _get("pain", 0.0),
        1.0 if patient.get("mrn") else 0.0
    ]])

    try:
        probabilities = triage_model.predict_proba(features)[0]
        sorted_probs = np.sort(probabilities)[::-1]
        
        top_class_idx = int(np.argmax(probabilities))
        predicted_esi = top_class_idx + 1 # Convert 0-indexed back to 1-5
        
        top_prob = float(sorted_probs[0])
        # Margin is the gap between the top choice and the second choice
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top_prob
        
        return predicted_esi, top_prob, margin
    except Exception as e:
        logger.error(f"XGBoost Inference error: {e}")
        return 4, 0.50, 0.0

def generate_deterministic_badges(patient: Dict[str, Any], routing: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Deterministically generates visual risk badges to prevent LLM hallucinations.
    Uses age-calibrated rules (Infant vs Adult).
    """
    badges = []
    is_pediatric = routing.get("model_route") == "PEDIATRIC_MODEL"
    
    # Blood Pressure Badges
    sys = patient.get("bpSys")
    if sys is not None:
        if sys < 90:
            badges.append({"label": f"Hypotension (BP {sys})", "type": "danger"})
        elif sys > 160:
            badges.append({"label": f"Hypertension (BP {sys})", "type": "warning"})
            
    # SpO2 Badges
    spo2 = patient.get("spo2")
    if spo2 is not None:
        if spo2 <= 92:
            badges.append({"label": f"Hypoxia (SpO2 {spo2}%)", "type": "danger"})
        elif spo2 >= 95:
            badges.append({"label": f"Normal O2 ({spo2}%)", "type": "success"})

    # Heart Rate Badges (Age-Calibrated)
    hr = patient.get("hr")
    if hr is not None:
        if is_pediatric:
            if hr > 180: 
                badges.append({"label": f"Pediatric Tachycardia ({hr})", "type": "danger"})
            elif hr < 160: 
                badges.append({"label": f"Normal Ped HR ({hr})", "type": "success"})
        else:
            if hr > 120: 
                badges.append({"label": f"Tachycardia ({hr})", "type": "danger"})
            elif 60 <= hr <= 100: 
                badges.append({"label": f"Normal HR ({hr})", "type": "success"})

    return badges

def calculate_dynamic_confidence(
    patient: Dict[str, Any], 
    llm_esi: int, 
    ml_esi: int,
    ml_prob: float,
    margin: float,
    complaint_category: str,
    routing_penalty: float
) -> int:
    """
    The Context-Aware Confidence Engine.
    Adjusts the true ML probability based on clinical context and algorithmic consensus.
    """
    # Start with the actual mathematical probability from XGBoost
    base_confidence = ml_prob * 100.0
    
    # 1. Routing Uncertainty (e.g., if age was guessed)
    base_confidence -= (routing_penalty * 100) 
    
    # 2. Consensus Modifier (Algorithmic Disagreement)
    if llm_esi != ml_esi:
        base_confidence -= 15.0 # LLM read text that overrides the numerical model
    else:
        # Boost slightly if models agree AND the mathematical margin is strong
        base_confidence += (margin * 10) 

    # 3. Context-Aware Missing Data Penalty
    has_bp = patient.get("bpSys") is not None
    has_temp = patient.get("temp") is not None
    has_spo2 = patient.get("spo2") is not None
    
    if complaint_category == "CARDIAC" and not has_bp:
        base_confidence -= 30.0 # High-Stakes Penalty
    elif complaint_category == "RESPIRATORY" and not has_spo2:
        base_confidence -= 30.0 # High-Stakes Penalty
    elif complaint_category == "TRAUMA_LOCALIZED":
        pass # Low-Risk Forgiveness (No penalty for missing vitals on a sprained ankle)
    else:
        # Standard penalties for generic complaints
        if not has_bp: base_confidence -= 10.0
        if not has_temp: base_confidence -= 5.0
        
    return max(0, min(99, int(base_confidence)))

def execute_triage_assessment(patient: dict, scrubbed_complaint: str, ehr_history: str, routing: dict):
    """
    Fuses the RAG context, calls the LLM, and calculates the final confidence score.
    """
    ml_esi, ml_prob, margin = run_real_xgboost_inference(patient)
    
    system_prompt = """
    You are an expert ER triage AI. Analyze vitals, complaint, and EHR history.
    Respond strictly in JSON format with exactly three keys:
    1. "category": Classify the complaint as exactly one of: ["CARDIAC", "TRAUMA_LOCALIZED", "RESPIRATORY", "OTHER"]
    2. "esi": Integer from 1 to 5 representing the recommended ESI level.
    3. "rationale": Max 2 sentences, under 40 words justifying the ESI. 
    """
    
    user_prompt = f"Vitals: HR {patient.get('hr')}, BP {patient.get('bpSys')}, SpO2 {patient.get('spo2')} | Complaint: {scrubbed_complaint} | EHR: {ehr_history}"

    try:
        response = llm_client.chat.completions.create(
            model="qwen/qwen3.8-27b" if settings.OPENAI_API_KEY.startswith("gsk_") else "gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        raw_content = response.choices[0].message.content or "{}"
        llm_data = json.loads(raw_content)
        
        confidence = calculate_dynamic_confidence(
            patient=patient,
            llm_esi=llm_data.get("esi", ml_esi),
            ml_esi=ml_esi,
            ml_prob=ml_prob,
            margin=margin,
            complaint_category=llm_data.get("category", "OTHER"),
            routing_penalty=routing.get("uncertainty_penalty", 0.0)
        )
        
        rationale = llm_data.get("rationale", "")
        # Dynamic Prompting/Injection: Force uncertainty to be front-and-center
        if confidence < 80:
            rationale = f"LOW CONFIDENCE ({confidence}%): Missing vital data or algorithmic disagreement detected. " + rationale

        return {
            "recommended_esi": llm_data.get("esi", ml_esi),
            "confidence_score": confidence,
            "rationale": rationale,
            "badges": generate_deterministic_badges(patient, routing),
            "ml_model_esi": ml_esi
        }
        
    except Exception as e:
        logger.error(f"Assessment Engine Error: {e}")
        return {
            "recommended_esi": ml_esi,
            "confidence_score": int(ml_prob * 100),
            "rationale": "SYSTEM ERROR: Relying on deterministic XGBoost fallback rules.",
            "badges": generate_deterministic_badges(patient, routing),
            "ml_model_esi": ml_esi
        }