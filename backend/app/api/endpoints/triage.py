from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.app.middleware.pii_scrubber import scrub_pii
from backend.app.services.rag_engine import retrieve_patient_history
from backend.app.services.routing_engine import route_patient_payload
from backend.app.services.assessment_engine import execute_triage_assessment

router = APIRouter()

class TriageRequest(BaseModel):
    mrn: Optional[str] = None
    name: str
    age: Optional[float] = None
    dob: Optional[str] = None
    complaint: str
    hr: Optional[int] = None
    bpSys: Optional[int] = None
    bpDia: Optional[int] = None
    spo2: Optional[int] = None
    temp: Optional[float] = None
    pain: Optional[int] = None

@router.post("/evaluate")
async def evaluate_patient(request: TriageRequest):
    try:
        # Step 1: Fail-Closed PII Scrubbing Middleware
        safe_complaint = scrub_pii(request.complaint, request.name)
        
        # Step 2: Stratified Age-Calibrated Routing
        patient_dict = request.model_dump()
        routing_info = route_patient_payload(patient_dict)
        
        # Step 3: Semantic RAG Retrieval with Metadata Hard-Filtering
        ehr_history = "NO PRIOR HOSPITAL HISTORY FOUND."
        if request.mrn:
            ehr_history = retrieve_patient_history(request.mrn, safe_complaint)
            
        # Step 4: Execute Multi-Class XGBoost + LLM Assessment + Dynamic Confidence Engine
        assessment = execute_triage_assessment(
            patient=patient_dict,
            scrubbed_complaint=safe_complaint,
            ehr_history=ehr_history,
            routing=routing_info
        )
        
        return {
            "status": "success",
            "routing": routing_info,
            "recommended_esi": assessment["recommended_esi"],
            "confidence_score": assessment["confidence_score"],
            "rationale": assessment["rationale"],
            "badges": assessment["badges"],
            "ml_model_esi": assessment["ml_model_esi"],
            "retrieved_history_snippet": (
                ehr_history[:120] + "..." 
                if ehr_history != "NO PRIOR HOSPITAL HISTORY FOUND." 
                else None
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))