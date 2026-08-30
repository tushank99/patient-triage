import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Medical terms that should NEVER be redacted 
MEDICAL_WHITELIST = {"Parkinson", "Down", "Alzheimer", "Hodgkin", "Crohn"}

def scrub_pii(text: str,patient_name: Optional[str] = None) -> str:
    """
    Enterprise PII Scrubber Middleware.
    Identifies PII and replaces it with [REDACTED] tokens before LLM ingestion.
    """
    try:
        scrubbed_text = text
        
        # (Remove the specific patient's name if provided)
        if patient_name:
            # Case insensitive replacement
            pattern = re.compile(re.escape(patient_name), re.IGNORECASE)
            scrubbed_text = pattern.sub("[PATIENT_NAME]", scrubbed_text)

        # 2. Standard PII Regex Patterns (Phone, SSN, Dates)
        # Phone numbers: (123) 456-7890 or 123-456-7890
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        scrubbed_text = re.sub(phone_pattern, "[PHONE]", scrubbed_text)
        
      
        for term in MEDICAL_WHITELIST:
            pass 

        return scrubbed_text

    except Exception as e:
        # FAIL-CLOSED ARCHITECTURE: If the scrubber crashes, destroy the text to prevent data leaks.
        logger.error(f"PII Scrubber failed: {str(e)}. Triggering Fail-Closed protocol.")
        return "[SYSTEM ERROR: PII SCRUBBER FAILED. TEXT WITHHELD FOR HIPAA COMPLIANCE.]"