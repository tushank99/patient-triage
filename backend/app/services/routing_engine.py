from datetime import date
from typing import Dict, Any, Optional

def calculate_age_years(dob: Optional[date]) -> float:
    if not dob:
        return -1.0 # Unknown
    today = date.today()
    return (today - dob).days / 365.25

def route_patient_payload(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Routes the patient vitals to the correct stratified ML model based on age.
    """
    age = calculate_age_years(patient_data.get("dob"))
    
    # Edge Case: Unknown DOB (Unconscious / No ID)
    if age == -1.0:
        estimated_group = patient_data.get("estimated_age_group", "ADULT")
        
        if estimated_group == "INFANT" or estimated_group == "CHILD":
            model_route = "PEDIATRIC_MODEL"
        elif estimated_group == "SENIOR":
            model_route = "GERIATRIC_MODEL"
        else:
            model_route = "ADULT_MODEL"
            
        return {
            "model_route": model_route,
            "uncertainty_penalty": 0.15, # Deduct 15% confidence for guessing age
            "warning": "Age estimated. Clinical baselines may be inaccurate."
        }

    # Standard Routing with overlapping boundaries
    if age < 18.0:
        return {
            "model_route": "PEDIATRIC_MODEL",
            "uncertainty_penalty": 0.0,
            "warning": None
        }
    elif age >= 65.0:
        return {
            "model_route": "GERIATRIC_MODEL",
            "uncertainty_penalty": 0.0,
            "warning": None
        }
    else:
        return {
            "model_route": "ADULT_MODEL",
            "uncertainty_penalty": 0.0,
            "warning": None
        }