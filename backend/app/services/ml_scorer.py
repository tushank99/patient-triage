def calculate_mock_ml_score(vitals: dict) -> dict:
    """
    Simulates an XGBoost Tabular model. 
    Starts at ESI 4 (Routine) and adjusts based on vital deviation.
    """
    base_esi = 4
    shap_drivers = []
    
    hr = vitals.get("heart_rate") or 80
    sbp = vitals.get("systolic_bp") or 120
    pain = vitals.get("pain_score") or 0

    if hr > 110:
        base_esi -= 1
        shap_drivers.append(f"Elevated Heart Rate ({hr} bpm)")
        
    if sbp > 160:
        base_esi -= 1
        shap_drivers.append(f"Hypertension (SBP {sbp})")
        
    if pain >= 7:
        if base_esi > 3:
            base_esi = 3
        shap_drivers.append(f"Severe Pain Score ({pain}/10)")

    # Ensure ESI stays between 1 and 5
    final_esi = max(1, min(5, base_esi))
    
    # If vitals are perfectly normal, note that.
    if not shap_drivers:
        shap_drivers.append("Vitals within normal limits")

    return {
        "ml_esi": final_esi,
        "shap_drivers": shap_drivers
    }
