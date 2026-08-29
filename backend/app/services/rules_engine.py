def evaluate_guardrails(age: float, vitals: dict, complaint: str) -> list:
    """
    Evaluates absolute heuristics. Returns a list of escalation flags.
    """
    flags = []
    
    # Safely extract vitals with defaults
    temp = vitals.get("temperature_celsius") or 37.0
    spo2 = vitals.get("oxygen_saturation") or 100.0
    hr = vitals.get("heart_rate") or 80
    
    complaint_lower = complaint.lower() if complaint else ""

    # 1. The Pediatric Fever Trap
    if age <= 0.25 and temp >= 38.0: # 3 months or younger
        flags.append({"level": 2, "reason": "Pediatric patient (<3mo) with fever >= 38.0C"})
        
    # 2. The Silent Critical (Hypoxia)
    if spo2 <= 92.0:
        flags.append({"level": 2, "reason": "Critical Hypoxia (SpO2 <= 92%)"})
        
    # 3. Extreme Vitals (ESI 1 - Immediate)
    if hr < 40 or hr > 150:
        flags.append({"level": 1, "reason": "Extreme abnormal heart rate detected"})
        
    # 4. Geriatric Fall Risk
    if age >= 65.0 and any(word in complaint_lower for word in ["fall", "fell", "head", "dizzy"]):
        flags.append({"level": 2, "reason": "Geriatric fall/head trauma protocol"})

    return flags
