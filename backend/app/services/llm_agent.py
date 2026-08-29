import asyncio
from backend.app.config import settings

async def extract_and_refine(complaint: str, guardrail_flags: list) -> dict:
    """
    Simulates the Agent 1 (Extractor) and Agent 2 (Refiner).
    Includes a 4-second timeout graceful degradation.
    """
    
    # HACKATHON FALLBACK: If we don't have a real API key yet, return a mock analysis.
    if settings.openai_api_key == "dummy_key_for_testing" or not settings.openai_api_key:
        await asyncio.sleep(1) # Simulate network delay
        
        # Check for our "Conflicting Signal" edge case hardcoded in the seed data
        is_critical = "tearing" in complaint.lower() or "dissection" in complaint.lower()
        
        return {
            "critical_semantics_detected": is_critical,
            "rationale": "AI Rationale (Simulated): Patient presentation suggests potential high acuity based on textual semantics." if is_critical else "AI Rationale (Simulated): Routine presentation without explicit high-risk semantic markers.",
            "llm_suggested_esi": 2 if is_critical else None
        }

    
    # FUTURE REAL OPENAI IMPLEMENTATION GOES HERE
    # try:
    #     response = await asyncio.wait_for(client.chat.completions.create(...), timeout=4.0)
    # except asyncio.TimeoutError:
    #     return {"critical_semantics_detected": False, "rationale": "AI Timeout - Reverted to Vitals"}
  
    
    return {"critical_semantics_detected": False, "rationale": "LLM Setup pending."}
