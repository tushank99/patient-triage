import chromadb
from typing import Optional
from openai import OpenAI
from backend.app.core.config import settings
import logging

logger = logging.getLogger(__name__)

chroma_client = chromadb.Client()
ehr_collection = chroma_client.get_or_create_collection(name="patient_ehr_history")
llm_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1" if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("gsk_") else None
)

def seed_mock_ehr_data():
    """Seeds the DB with dates to handle 'The Temporal Trap'"""
    if ehr_collection.count() > 0: return
        
    mock_records = [
        {
            "id": "doc1",
            "mrn": "MRN-A3F9", # Exact MRN for Metadata Partitioning
            "text": "[DATE: Oct 2024] Discharge Summary: Severe hypertension, Type 2 diabetes. Prior mild MI in 2022. Allergic to Penicillin."
        },
        {
            "id": "doc2",
            "mrn": "MRN-A3F9",
            "text": "[DATE: Jan 2021] ER Note: Patient arrived in 3rd trimester of pregnancy. (RESOLVED)"
        }
    ]
    
    ehr_collection.add(
        ids=[record["id"] for record in mock_records],
        documents=[record["text"] for record in mock_records],
        # CRITICAL: We store the MRN as metadata for hard-filtering
        metadatas=[{"mrn": record["mrn"]} for record in mock_records]
    )

def retrieve_patient_history(mrn: str, complaint: str) -> str:
    """
    RAG Retrieval with 'Identity Mix-Up' prevention.
    """
    try:
        # We query by the semantic complaint, BUT hard-filter strictly by MRN
        results = ehr_collection.query(
            query_texts=[complaint],
            n_results=3,
            where={"mrn": mrn} # Prevents Identity Mix-Up edge case
        )
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            # Combine the top notes into a single string
            retrieved_notes = "\n".join(results['documents'][0])
            logger.info(f"RAG: Found relevant history for {mrn}")
            return retrieved_notes
        else:
            return "NO PRIOR HOSPITAL HISTORY FOUND. First-time walk-in patient."
            
    except Exception as e:
        logger.error(f"RAG Retrieval Error: {e}")
        return "NO PRIOR HOSPITAL HISTORY FOUND."

def generate_ai_assessment(patient_data: dict, scrubbed_complaint: str, ehr_history: str):
    """
    LLM generation handling the Contradictory Data and Temporal Trap edge cases.
    """
    system_prompt = """
    You are an expert ER triage AI. Analyze vitals, complaint, and retrieved EHR history.
    
    CLINICAL RULES:
    1. If the EHR contains 'NO PRIOR HOSPITAL HISTORY', evaluate strictly on current vitals and complaint.
    2. THE TEMPORAL TRAP: If a retrieved note is dated over 9 months ago and involves a temporary condition (pregnancy, acute infection), treat it as resolved unless current symptoms suggest otherwise.
    3. CONTRADICTION RULE: If the patient's dictated complaint contradicts documented allergies in the EHR, prioritize the EHR as a clinical warning.
    
    OUTPUT FORMAT:
    Extract 2-3 risk factors (bullet points).
    Write a 2-3 sentence clinical rationale explaining the suggested ESI score.
    """
    
    user_prompt = f"""
    Age: {patient_data.get('age')}
    Vitals: HR {patient_data.get('hr')}, BP {patient_data.get('bpSys')}/{patient_data.get('bpDia')}, SpO2 {patient_data.get('spo2')}%, Temp {patient_data.get('temp')}C
    Complaint: {scrubbed_complaint}
    EHR History: {ehr_history}
    """
    # ... call llm_client.chat.completions.create() as before